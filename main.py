import torch

import detector
from super_gradients.training import models
import os
from tqdm import tqdm
import cv2

TEST_DIRECTORY = 'extract/vids/'
OUTPUT_TEST = 'results/vids/'

TEST_IMAGE_DIR = 'extract/frames/'
RESULT_IMAGE_DIR = 'results/images/'

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # setup_device(device='cuda')

    ## Train model
    # detector.model_training()

    ## Declare model
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    model = models.get(
        model_name='yolo_nas_s',
        checkpoint_path='checkpoints/yolo_nas_s/1_run/ckpt_best.pth',
        num_classes=1
    ).to(device)
    model.eval()

    ## Run model on images
    ROOT_TEST = "YN-HT-s/images/test"
    all_images = os.listdir(ROOT_TEST)
    for image in tqdm(all_images, total=len(all_images)):
        image_path = os.path.join(ROOT_TEST, image)
        out = model.predict(image_path)
        out.save('results/images')
        os.rename(
            'results/images/pred_0.jpg',
            os.path.join('results/images/1_run/', image)
        )

    ## Detect all objects in given video
    # cap = cv2.VideoCapture('extract/vids/Berghouse Leopard Jog.mp4')
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter('results/vids/Berghouse Leopard Jog.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))
    #
    # while cap.isOpened():
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #
    #     # Color correct
    #     image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #
    #     # Perform detection
    #     results = model.predict(image_rgb)
    #     results.save(OUTPUT_TEST)
    #
    #     # Draw boxes and labels on the frame
    #     annotated_frame = cv2.imread(f"{OUTPUT_TEST}pred_0.jpg")
    #
    #     # Write the frame to the output video
    #     out.write(annotated_frame)
    #
    #     # Optional: Display the frame
    #     cv2.imshow('Frame', annotated_frame)
    #     if cv2.waitKey(1) == ord('q'):
    #         break
    #
    # # Release resources
    # cap.release()
    # out.release()
    # cv2.destroyAllWindows()

