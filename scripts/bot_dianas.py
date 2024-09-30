import subprocess
import os
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, CallbackContext
import cv2
import numpy as np

# Function to calculate the score based on the position of the hole NOT WORKING
def calculate_score(center, x, y):
    distance = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    if distance <= 25:
        return 10
    elif distance <= 50:
        return 9
    elif distance <= 75:
        return 8
    elif distance <= 100:
        return 7
    elif distance <= 125:
        return 6
    elif distance <= 150:
        return 5
    elif distance <= 175:
        return 4
    elif distance <= 200:
        return 3
    elif distance <= 225:
        return 2
    elif distance <= 250:
        return 1
    else:
        return 0

# Function to process the results from YOLOv5
def process_results(center, coords_path):
    with open(coords_path, 'r') as f:
        lines = f.readlines()

    results = []
    for line in lines:
        parts = line.split()
        class_id = int(parts[0])
        x_min = float(parts[1]) - float(parts[3]) / 2
        y_min = float(parts[2]) - float(parts[4]) / 2
        x_max = float(parts[1]) + float(parts[3]) / 2
        y_max = float(parts[2]) + float(parts[4]) / 2
        results.append((class_id, x_min, y_min, x_max, y_max))

    # Calculate the scores
    total_scores = 0
    for res in results:
        x1, y1, x2, y2 = res[1], res[2], res[3], res[4]
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        score = calculate_score(center, x_center, y_center)
        total_scores += score

    return total_scores

# Function to process the image with YOLOv5
async def process_image(update: Update, context: CallbackContext):
    # Check if a photo was received
    if not (update.message.document or update.message.photo):
        await update.message.reply_text("Please send an image.")
        return

    file_path = "./image.jpg"  # Path where the image will be saved

    try:
        if update.message.document:
            file_info = await update.message.document.get_file()
            await file_info.download_to_drive(file_path)

        elif update.message.photo:
            photo = update.message.photo[-1]
            file_info = await photo.get_file()
            await file_info.download_to_drive(file_path)

        # Run YOLOv5
        yolo_executable_path = "../yolov5/detect.py"
        yolo_params = [
            "--source", file_path,
            "--weights", "../yolov5/best.pt",
            "--img", "640",
            "--hide-labels",
            "--exist-ok",
            "--project", "../outputs",
            "--line-thickness", "1",
            "--save-txt"
        ]
        command = ["python", yolo_executable_path] + yolo_params
        result = subprocess.run(command, capture_output=True, text=True)

        if result.returncode != 0:
            print(result.stderr)
            await update.message.reply_text("Error processing the image with YOLOv5.")
            return

        processed_image_path = f"../outputs/{os.path.basename(file_path)}"
        if os.path.exists(processed_image_path):
            with open(processed_image_path, 'rb') as img_file:
                await context.bot.send_photo(chat_id=update.effective_chat.id, photo=img_file)

            # Process results
            total_score = process_results((640 // 2, 640 // 2), f"../outputs/labels/{os.path.splitext(os.path.basename(file_path))[0]}.txt")
            await context.bot.send_message(chat_id=update.effective_chat.id, text=f"Total score: {total_score}")
        else:
            await update.message.reply_text("Sorry, the image could not be processed correctly.")

    except Exception as e:
        await update.message.reply_text("An error occurred: " + str(e))

# Start function
async def start(update: Update, context: CallbackContext):
    await update.message.reply_text("Hello! Send me an image and I'll send you the processed image.")

# Main function
def main():
    token = ""  # Replace this with your token

    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO | filters.ALL, process_image))

    # Start the event loop
    app.run_polling()

if __name__ == '__main__':
    main()
