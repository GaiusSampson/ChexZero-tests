import os
import argparse
from tqdm import tqdm
import torch
from torch import nn
import torch.optim as optim
from train import load_data, load_clip, preprocess_text
from torch.utils.tensorboard import SummaryWriter
from zero_shot import make_true_labels, run_softmax_eval
from eval import evaluate
from load_test_data import load_test_data
import numpy as np
from zero_shot_biobert import make_true_labels, run_softmax_eval


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cxr_filepath', type=str, default='/data/cxr.h5', help="Directory to load chest x-ray image data from.")
    parser.add_argument('--txt_filepath', type=str, default='/data/mimic_impressions.csv', help="Directory to load radiology report impressions text from.")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save_interval', type=int, default=1000)
    parser.add_argument('--log_interval', type=int, default=200)
    parser.add_argument('--save_dir', type=str, default="checkpoints/", help="Directory to save the trained model.")
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--optimizer', type=str, default="sgd")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--rho', type=float, default=0.5)
    parser.add_argument('--eta', type=float, default=0.01)
    parser.add_argument('--context_length', type=int, default=77)
    parser.add_argument('--random_init', action='store_true')
    parser.add_argument('--model_name', type=str, default="pt-imp")
    parser.add_argument('--run_dir', type=str, default=None)
    parser.add_argument('--base_model', type=str, default="SWINV2", help="The pretrained CLIP base model")
    parser.add_argument('--use_biobert', action='store_true',
                        help='Use BioClinicalBERT for text encoder')
    parser.add_argument('--biobert_model', type=str,
                        default='emilyalsentzer/Bio_ClinicalBERT',
                        choices=[
                            'emilyalsentzer/Bio_ClinicalBERT',
                            'emilyalsentzer/Bio_Discharge_Summary_BERT',
                            'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
                            'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
                        ],
                        help='BioBERT model variant to use')
    parser.add_argument('--flava', help="Finetune on FLAVA instead on CLIP", action='store_true')
    args = parser.parse_args()
    return args

def model_pipeline(config, verbose=0): 
    # set up logging
    if config.run_dir is None:
        config.run_dir = f"./runs/{config.model_name}"
    writer = SummaryWriter(config.run_dir)
    
    # make the model, data, and optimization problem
    model, data_loader, device, criterion, optimizer, minimizer = make(config)

    # and use them to train the model
    train(model, data_loader, device, criterion, optimizer, config, writer, minimizer)

    # save model
    model_path = os.path.join(config.save_dir, str(config.model_name), 'checkpoint.pt')
    save(model, model_path)

    if verbose: 
        print(model)
    return model

def make(config): 
    pretrained = not config.random_init
    data_loader, device = load_data(
        config.cxr_filepath, 
        config.txt_filepath, 
        batch_size=config.batch_size, 
        pretrained=pretrained, 
        column="impression"
    )
    
    if config.base_model != "SWINV2":
        model = load_clip(
            model_path=None, 
            pretrained=pretrained, 
            context_length=config.context_length, 
            pretrained_model=config.base_model
        )
    else:
        model = load_clip(
                model_path=None,
                pretrained=False,
                context_length=args.context_length,
                swin_encoder=True,
                use_biobert=args.use_biobert,
                biobert_model=args.biobert_model
        )
    model = model.float()
    model.to(device)
    model.train()
    print('Model on Device.')
    
    parameters = model.parameters()
    
    # make the optimizer
    minimizer = None
    criterion = nn.CrossEntropyLoss().cuda()
    if config.optimizer == "adam": 
        optimizer = optim.AdamW(parameters, lr=config.lr)
    elif config.optimizer == "sgd": 
        optimizer = optim.SGD(parameters, lr=config.lr, momentum=config.momentum)
     
    return model, data_loader, device, criterion, optimizer, minimizer

def accuracy(y_true, y_pred, threshold=0.5):
    y_pred_binary = (y_pred >= threshold).astype(int)
    correct_preds = np.sum(y_true == y_pred_binary)
    total_preds = y_true.size
    
    return correct_preds / total_preds


def train(model, loader, device, criterion, optimizer, config, writer, minimizer=None): 
    model_save_dir = os.path.join(config.save_dir, config.model_name)
    if not os.path.exists(model_save_dir): 
        # Create a new folder if not exists
        os.makedirs(model_save_dir)
    
    # Run training
    total_batches = len(loader) * config.epochs
    example_ct = 0  # number of examples seen
    batch_ct = 0
    report_freq = config.log_interval
    highest_val_auc = 0 # save highest mean auc
    
    # load validation labels
    #----------------------------------------
    cxr_labels = ['Atelectasis','Cardiomegaly',
                  'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
                  'Lung Opacity', 'No Finding','Pleural Effusion', 'Pleural Other', 'Pneumonia', 
                  'Pneumothorax', 'Support Devices']

    cxr_pair_template = ("{}", "no {}")
    test_loader = load_test_data("test_data/chexpert_val.h5")
    y_true = make_true_labels("test_data/val.csv", cxr_labels)
    #----------------------------------------
    
    best_models = {i:{"Filepath": os.path.join(model_save_dir, f"Best_{i}.pt"), "Mean_AUC": 0} for i in range(10)}
  
    for epoch in range(config.epochs):
        running_loss = 0.0 # running loss over batch

        for data in tqdm(loader):
            # get the images
            images = data['img']
            texts = data['txt']
            
            if not config.flava:
                texts = preprocess_text(texts, model) 
                
            # perform step for a single batch
            loss = train_batch(config, images, texts, model, device, criterion, optimizer, minimizer)
            
            example_ct +=  len(images)
            batch_ct += 1
            running_loss += loss.item()
            
            # Report metrics every `report_freq` batch
            if (batch_ct % report_freq) == 0:
                train_log(running_loss / report_freq, example_ct, batch_ct, epoch, writer)
                running_loss = 0.0
            
            if (batch_ct % config.save_interval) == 0:
                
                # Evaluate
                cxr_pair_template = ("{}", "no {}")
                model = model.cpu()
                model.eval()
                y_pred = run_softmax_eval(model, test_loader, cxr_labels, cxr_pair_template)
                model = model.to(device)
                stats = evaluate(y_pred, y_true, cxr_labels)
                mean_auc = stats.mean(1)[0]
                model.train()
                
                # used for keeping track of best models
                smallest_score = 101
                smallest_index = 0
                for k,v in best_models.items():
                    if v["Mean_AUC"] < smallest_score:
                        smallest_score = v["Mean_AUC"]
                        smallest_index = k

                if mean_auc > smallest_score:
                    print(f'Validation score in top 10, saving to {best_models[smallest_index]["Filepath"]}')
                    print(f"Validation score: {mean_auc}")
                    best_models[smallest_index]["Mean_AUC"] = mean_auc
                    save(model, best_models[smallest_index]["Filepath"])

                else:
                    print(f'Validation score not in top 10')
                    print(f"Validation score: {mean_auc}")

                    
            del images, texts
            torch.cuda.empty_cache()

def train_batch(config, images, texts, model, device, criterion, optimizer, minimizer=None):
    if not config.flava:
        if model.use_biobert:
        # texts is already a dict on device from preprocess_text
            images = images.to(device)
        else:
            images, texts = images.to(device), texts.to(device)
        batch_size = images.shape[0]
    else:
        images = [img for img in images]
        batch_size = len(images)
        
        images = model.image_processor(images, return_tensors="pt").to(device)
        texts = model.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
        
    # Forward pass ➡
    logits_per_image, logits_per_text = model(images, texts)

    # Create labels
    
    labels = torch.arange(batch_size).to(device)
    
    # Compute loss
    loss_img = criterion(logits_per_image, labels)
    loss_txt = criterion(logits_per_text, labels)
    loss = (loss_img + loss_txt)/2 # avg. img and txt loss

    # Backward pass ⬅
    loss.backward()
    
    if (minimizer != None):
        # using ASAM, need to ascent and descent
        minimizer.ascent_step()
        
        loss_temp = loss
        logits_per_image, logits_per_text = model(images, texts)
        
        loss_img = criterion(logits_per_image, labels)
        loss_txt = criterion(logits_per_text, labels)
        loss = (loss_img + loss_txt)/2
        
        loss.backward()
        minimizer.descent_step()
        
        # return what the loss was before optimizer uses it
        loss = loss_temp
        
    else:
        # Step with optimizer
        optimizer.step()
        
    optimizer.zero_grad()
    return loss

def train_log(loss, example_ct, batch_ct, epoch, writer):
    loss = float(loss)
    print(f"Loss after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}")
    writer.add_scalar("Loss/batch", loss, batch_ct)
    
def save(model, path): 
    torch.save(model.state_dict(), path)
    
if __name__ == "__main__":
    args = parse_args()
    model = model_pipeline(args)
    

