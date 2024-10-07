import ujson as json
from datetime import datetime


def save_results(
    model_name: str,
    positive_results: tuple[any],
    partitial_results: tuple[any],
    negative_results: tuple[any],
):
    result_save_path = f"D:/MTCNN/verify/results/{model_name}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.json"
    with open(result_save_path, "w", encoding="utf-8") as file:
        data = {
            "positive_result": [str(result.numpy()) for result in positive_results],
            "partitial_result": [str(result.numpy()) for result in partitial_results],
            "negative_result": [str(result.numpy()) for result in negative_results],
        }
        json.dump(data, file)
