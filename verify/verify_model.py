import sys

sys.path.append(r"D:\mtcnn\verify\func")
from func.get_dataset import get_dataset
from func.preprocess_dataset import preprocess_dataset
from func.verify import verify
from func.save_results import save_results


def verify_model(imageset_name: str, model_name: str, model_path: str):
    positive, partitial, negative = get_dataset(imageset_name, model_name)
    positive_image_tensors, positive_label_tensors = preprocess_dataset(
        model_name, positive
    )
    partitial_image_tensors, partitial_label_tensors = preprocess_dataset(
        model_name, partitial
    )
    negative_image_tensors, negative_label_tensors = preprocess_dataset(
        model_name, negative
    )
    positive_results = verify(
        model_name, model_path, positive_image_tensors, positive_label_tensors
    )
    partitial_results = verify(
        model_name, model_path, partitial_image_tensors, partitial_label_tensors
    )
    negative_results = verify(
        model_name, model_path, negative_image_tensors, negative_label_tensors
    )
    save_results(model_name, positive_results, partitial_results, negative_results)
    print(f"positive_result: {positive_results}")
    print(f"partitial_result: {partitial_results}")
    print(f"negative_result: {negative_results}")
