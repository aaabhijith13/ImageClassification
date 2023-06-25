#TODO: Import your dependencies.
import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets, transforms
from torch import nn, optim
import os
import argparse

# from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True


#TODO: Import dependencies for Debugging andd Profiling
from smdebug import modes
import smdebug.pytorch as smd

def test(model, test_loader, criterion, device, hook):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    print("Testing Model on Whole Testing Dataset")    
    model.eval()
    hook.set_mode(modes.EVAL)
    running_loss=0.0
    running_corrects=0.0
    
    for inputs, labels in test_loader:
        inputs=inputs.to(device)
        labels=labels.to(device)        
        outputs=model(inputs)
        loss=criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += float(loss.item() * inputs.size(0))
        running_corrects += float(torch.sum(preds == labels.data))

    total_loss = float(running_loss) // float(len(test_loader.dataset))
    total_acc = float(running_corrects) // float(len(test_loader.dataset))
    
    # here works regexp for "Testing Loss"
    print(f"Testing Loss: {total_loss}")
    print(f"Testing Accuracy: {total_acc}")


def train(model, train_loader, valid_loader, criterion, optimizer, args, device, hook):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    epochs=args.epochs
    best_loss=float(1e6)
    image_dataset={'train':train_loader, 'valid':valid_loader}
    loss_counter=0
    
    for epoch in range(epochs):
        for phase in ['train', 'valid']:
            print(f"Epoch: {epoch}, Phase: {phase}")
            if phase=='train':
                model.train()
                hook.set_mode(modes.TRAIN)    
            else:
                model.eval()
                hook.set_mode(modes.EVAL)    
            running_loss = 0.0
            running_corrects = 0.0
            running_samples=0
            
            total_samples_in_phase = len(image_dataset[phase].dataset)

            for inputs, labels in image_dataset[phase]:
                inputs=inputs.to(device)
                labels=labels.to(device)                  
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase=='train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += float(loss.item() * inputs.size(0))
                running_corrects += float(torch.sum(preds == labels.data))
                running_samples+=len(inputs)

                accuracy = float(running_corrects)/float(running_samples)
                print("Epoch {}, Phase {}, Images [{}/{} ({:.0f}%)] Loss: {:.2f} Accuracy: {}/{} ({:.2f}%)".format(
                        epoch,
                        phase,
                        running_samples,
                        total_samples_in_phase,
                        100.0 * (float(running_samples) / float(total_samples_in_phase)),
                        loss.item(),
                        running_corrects,
                        running_samples,
                        100.0*accuracy,
                    ))
                 
                #NOTE: Comment lines below to train and test on whole dataset
                if (running_samples>(0.1*total_samples_in_phase)):
                    break
                
                
            epoch_loss = float(running_loss) // float(running_samples)
            epoch_acc = float(running_corrects) // float(running_samples)
            
            if phase=='valid':
                if epoch_loss<best_loss:
                    best_loss=epoch_loss
                else:
                    loss_counter+=1


            print('{} loss: {:.4f}, acc: {:.4f}, best loss: {:.4f}'.format(phase,
                                                                           epoch_loss,
                                                                           epoch_acc,
                                                                           best_loss))
            
        # Break training when loss starts to increase. I am not sure if this is required since training should handle these issues.
        if loss_counter==1:
            print("Finish training because epoch loss increased")            
            break
    return model

def net():
    # Load the pretrained ResNet model
    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False   

    model.fc = nn.Sequential(
                   nn.Linear(2048, 128),
                   nn.ReLU(inplace=True),
                   nn.Linear(128, 133))
    return model

def create_data_loaders(data, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''

    training_transform = transforms.Compose([
     transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ])

    testing_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        ])

    trainset = torchvision.datasets.ImageFolder(root= os.path.join(data, 'train'), transform=training_transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    validset = torchvision.datasets.ImageFolder(root= os.path.join(data, 'valid'), transform=testing_transform)
    valid_loader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=True)
    
    testset = torchvision.datasets.ImageFolder(root= os.path.join(data, 'test'), transform=testing_transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

    return train_loader, valid_loader, test_loader


def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device {device}")
    model=net()
    model.to(device)
    
    '''
    TODO: Create your loss and optimizer
    '''
    criterion = nn.CrossEntropyLoss(ignore_index=133)
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)

    hook = smd.Hook.create_from_json_file()
    hook.register_hook(model)
    hook.register_loss(criterion)  
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    print("Loading Data")

    train_loader, valid_loader, test_loader = create_data_loaders(args.data_path, args.batch_size)
    
    print("Loading Done, Model Training")
    model=train(model, train_loader, valid_loader, criterion, optimizer, args, device, hook)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    print("Training done, testing")
    test(model, test_loader, criterion, device, hook)
    '''
    TODO: Save the trained model
    '''
    print("Saving Model")
    torch.save(model.cpu().state_dict(), os.path.join(args.model_dir, "model.pth"))


if __name__=='__main__':
    '''
    TODO: Specify any training args that you might need
    '''
    parser=argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description="Project 3")
    parser.add_argument("--batch-size",type=int,default=50,metavar="N")
    parser.add_argument("--epochs",type=int,default=2,metavar="N")
    parser.add_argument("--lr", type=float, default=1.0, metavar="LR")
    parser.add_argument('--data_path', type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])  
    args = parser.parse_args()
    print("The arguments are: {}".format(args))
    main(args)
