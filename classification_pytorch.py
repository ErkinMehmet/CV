import torch,os
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms,models
from PIL import Image
import matplotlib.pyplot as plt
# data transformations of rdata augmentation and normalization
data_transforms={
    'train':transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val':transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# define the data dir
data_dir='/content/flowers_dataset'
image_datasets={x:datasets.ImageFolder(os.path.join(data_dir,x),data_transforms[x]) for x in ['train','val']}
dataloaders={x:torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in ['train','val']}
dataset_sizes={x:len(image_datasets[x]) for x in ['train','val']}
print(dataset_sizes)
class_names=image_datasets['train'].classes
print(class_names)

# load the pre-trained model
model=models.resnet18(pretrained=True)

# freeze all the layers except the final classifier layer
for name, param in model.named_parameters():
    if "fc" not in name:
        param.requires_grad=False
    else:
        param.requires_grad=True

# define the loss function and optimizer
criterion=nn.CrossEntropyLoss()
optimizer=optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9) # use all params

# move the model to GPU if available
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model=model.to(device)

# training loop
num_epochs=2
for epoch in range(num_epochs):
    for phase in ['train','val']:
        if phase=='train':
            model.train()
        else:
            model.eval()
        running_loss=0.0
        running_corrects=0
        for inputs, labels in dataloaders[phase]:
            inputs=inputs.to(device)
            labels=labels.to(device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(phase=='train'):
                outputs=model(inputs)
                _,preds=torch.max(outputs,1) # values, indexes
                loss=criterion(outputs,labels)
                if phase=='train': # back propagation + optimization only in train phase
                    loss.backward()
                    optimizer.step()
            running_loss+=loss.item()*inputs.size(0) # average loss per batch multiplied by batch size => total loss for the batch
            running_corrects+=torch.sum(preds==labels.data)
        epoch_loss=running_loss/dataset_sizes[phase] # average loss for the epoch (divided by number of samples)
        epoch_acc=running_corrects.double()/dataset_sizes[phase]
        print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

print("Training complete!")
torch.save(model.state_dict(), 'model/flower_model.pth')

# classification on unseen image
model=models.resnet18(pretrained=True)
model.fc=nn.Linear(model.fc.in_features, 1000) # adjust the final layer, but with the wrong number of output classes
model.load_state_dict(torch.load('model/flower_model.pth')) # load the trained weights, the weights for the final layer will not match
model.eval() # set the model to evaluation mode

# create a new model with the correct final layer then we load weights; we could just load the weights directly to the right model
new_model=models.resnet18(pretrained=True)
new_model.fc=nn.Linear(new_model.fc.in_features, 2)   # Adjust to match the desired output units
new_model.fc.weight.data=model.fc.weight.data[0:2]  # Copy weights for the first two classes
new_model.fc.bias.data=model.fc.bias.data[0:2]      # Copy bias for the first two classes

# load the preprocess the unseen image
image_path='resources/flower.jpg'
image=Image.open(image_path)
preprocess=transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
input_tensor=preprocess(image) 
input_batch=input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

# perform inference using the model
with torch.no_grad():
    output=new_model(input_batch)
_,pred=output.max(1)
print(f'Predicted class: {class_names[pred.item()]}')

# Display the image with the predicted class name
image = np.array(image)
plt.imshow(image)
plt.axis('off')
plt.text(10, 10, f'Predicted: {class_names[pred.item()]}', fontsize=12, color='white', backgroundcolor='red')
plt.show()