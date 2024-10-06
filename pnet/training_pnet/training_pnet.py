import sys

sys.path.append(r"D:\mtcnn\pnet/training_pnet\func")
from func.labels_generate import labels_generate
from func.dataset_generate import dataset_generate
from func.create_model import create_model
from func.callbacks import get_callback


def training_pnet(
    dataset_name, split_count=32, batch_size=64, max_epochs=100, model_path="auto"
):
    (
        train_image_paths,
        train_class_labels,
        train_bbox_labels,
        val_image_paths,
        val_class_labels,
        val_bbox_labels,
    ) = labels_generate(dataset_name)
    training_datasets = dataset_generate(
        train_image_paths,
        train_class_labels,
        train_bbox_labels,
        batch_size,
        split_count,
    )
    validation_dataset = dataset_generate(
        val_image_paths, val_class_labels, val_bbox_labels, batch_size, 1
    )
    pnet_model = create_model((12, 12, 3), model_path)
    callbacks = get_callback()

    for epoch in range(max_epochs):
        idx = epoch % len(training_datasets)
        print(f"\nepoch {epoch + 1}: training dataset[{idx}]")
        pnet_model.fit(
            training_datasets[idx],
            validation_data=validation_dataset,
            epochs=1,
            callbacks=callbacks,
        )
