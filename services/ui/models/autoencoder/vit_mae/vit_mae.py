from transformers import AutoImageProcessor, ViTMAEModel


def get_model(checkpoint):
    model = ViTMAEModel.from_pretrained(checkpoint)

    return model

    # image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
    # model = ViTMAEModel.from_pretrained("facebook/vit-mae-base")
    #
    # inputs = image_processor(images=image, return_tensors="pt")
    # outputs = model(**inputs)
    # last_hidden_states = outputs.last_hidden_state
