import subprocess
import os
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, CallbackContext, CallbackQueryHandler
import math
import cv2
import numpy as np

# Function to calculate the score based on the position of the hole
def calculate_score(center, x, y):
    distance = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
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

    total_scores = 0
    for res in results:
        x1, y1, x2, y2 = res[1], res[2], res[3], res[4]
        x_center = float((x1 + x2) * 1000 // 2)
        y_center = float((y1 + y2) * 1000 // 2)
        score = calculate_score(center, x_center, y_center)
        total_scores += score

    return total_scores

# Function to process the image with YOLOv5
async def process_image(update, context):
    if not (update.message.document or update.message.photo):
        await update.message.reply_text("Please send an image.")
        return

    file_path = None  # Initialize file_path here

    if update.message.document:
        file_info = update.message.document
        file_id = file_info.file_id
        file = await context.bot.get_file(file_id)
        file_path = f"./{file_id}.jpg"  # Use a consistent path
        await file.download_to_drive(file_path)

    if update.message.photo:
        photo = update.message.photo[-1]
        file_id = photo.file_id
        file = await context.bot.get_file(file_id)
        file_path = f"./{file_id}.jpg"  # Again, consistent path
        await file.download_to_drive(file_path)

    if not file_path:
        await update.message.reply_text("Could not retrieve the image.")
        return

    # YOLO processing part
    yolo_executable_path = "../yolov5/detect.py"
    yolo_params = [
        "--source", file_path, "--weights", "../yolov5/best.pt", "--img", "640",
        "--data", "../yolov5/data/custom_data.yaml", "--hide-labels",
        "--exist-ok", "--project", "../outputs", "--line-thickness", "1",
        "--save-txt"
    ]
    command = ["python", yolo_executable_path] + yolo_params
    subprocess.run(command)

    processed_image_path = f"../outputs/{os.path.basename(file_path)}"  # Ensure correct path
    if os.path.exists(processed_image_path):
        await context.bot.send_photo(chat_id=update.effective_chat.id, photo=open(processed_image_path, 'rb'))
        image = cv2.imread(file_path)
        total_score = process_results(
            (image.shape[1] // 2, image.shape[0] // 2),
            f"../outputs/labels/{os.path.splitext(os.path.basename(file_path))[0]}.txt"
        )
        await context.bot.send_message(chat_id=update.effective_chat.id, text=f"Total score: {total_score}")

        # Confirmation buttons
        keyboard = [
            [
                InlineKeyboardButton("✅", callback_data='confirm'),
                InlineKeyboardButton("❌", callback_data='deny')
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text("Is the detection correct?", reply_markup=reply_markup)
        context.user_data['file_path'] = file_path
        context.user_data['processed_image_path'] = processed_image_path
    else:
        await update.message.reply_text("Sorry, the image could not be processed correctly.")

# Function to handle confirmation from the user
async def confirm(update, context):
    query = update.callback_query
    await query.answer()  # Make sure to await this call

    if 'file_path' not in context.user_data or 'processed_image_path' not in context.user_data:
        await query.edit_message_text(text="No image found to confirm.")
        return

    if query.data == 'confirm':
        file_path = context.user_data['file_path']
        processed_image_path = context.user_data['processed_image_path']

        # Move confirmed images and their annotations to the training folder
        new_image_path = f"../yolov5/train_data/images/train/{os.path.basename(file_path)}"
        new_label_path = f"../yolov5/train_data/labels/train/{os.path.basename(file_path).replace('.jpg', '.txt')}"
        try:
            os.rename(file_path, new_image_path)
            os.rename(f"../outputs/labels/{os.path.basename(file_path).replace('.jpg', '.txt')}", new_label_path)

            await query.edit_message_text(text="Detection has been confirmed and added to the training set.")
            # Optionally, you can call your retraining script here
            # retrain_model()
        except FileExistsError:
            await query.edit_message_text(text="The file already exists in the training folder.")

    else:
        await query.edit_message_text(text="Detection not confirmed.")

    del context.user_data['file_path']
    del context.user_data['processed_image_path']

# Function to retrain the model
def retrain_model():
    train_script_path = "../yolov5/train.py"
    train_params = [
        "--data", "../yolov5/train_data/custom_data.yaml",
        "--weights", "../yolov5/best.pt",
        "--name", "telegram",
        "--hyp", "../yolov5/train_data/hyp.yaml",
        "--img", "640", 
        "--epochs", "3000",
        "--batch", "16",
        "--nosave",
        "--cache",
        "--patience", "400"
    ]
    command = ["python", train_script_path] + train_params
    subprocess.run(command)

# Start command handler
def start(update, context):
    update.message.reply_text("Hello! Send me an image and I'll send you the processed image.")

# Main function
def main():
    token = ""  # Replace this with your token
    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO | filters.ALL, process_image))
    app.add_handler(CallbackQueryHandler(confirm))

    # Start the event loop
    app.run_polling()

if __name__ == '__main__':
    main()
