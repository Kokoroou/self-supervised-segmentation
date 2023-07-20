from PIL import Image
from transformers import AutoImageProcessor, ViTMAEModel

if __name__ == "__main__":
    path = "data/processed/PolypGen2021_MultiCenterData_v3/positive/images/C1_104OLCV1_100H0002.jpg"
    image = Image.open(path)

    image_processor = AutoImageProcessor.from_pretrained("kokoroou/vit-mae-large-1")
    model = ViTMAEModel.from_pretrained("kokoroou/vit-mae-large-1")

    inputs = image_processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    print(outputs.last_hidden_state.shape,
          outputs.mask.shape,
          outputs.ids_restore.shape,)

    print(outputs.ids_restore)
