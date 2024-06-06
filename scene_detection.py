import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F

# Load pre-trained indoor/outdoor classification model (VGG16-hybrid1365)
model_file = 'vgg16_hybrid1365.pth'
model = torch.load(model_file, map_location=torch.device('cpu'))
model.eval()

# Define image preprocessing transformation
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def classify_indoor_outdoor(frame):
    image = Image.fromarray(frame)
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_batch)
        output = F.softmax(output, dim=1)
    
    # Define indoor and outdoor categories
    indoor_categories = range(151, 365)  # categories 151 to 365 are indoor
    outdoor_categories = range(0, 151)   # categories 0 to 150 are outdoor
    
    indoor_score = output[0, indoor_categories].sum().item()
    outdoor_score = output[0, outdoor_categories].sum().item()
    
    return indoor_score > outdoor_score
