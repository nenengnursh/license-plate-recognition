import os
import cv2
import yaml
import numpy as np
import matplotlib.gridspec as gridspec

from sort.sort import *
from ultralytics import YOLO
from keras.models import model_from_json
from werkzeug.utils import secure_filename
from sklearn.preprocessing import LabelEncoder
from utilall import get_vehicle, read_license_plate
from flask import Flask, render_template, url_for, redirect, request

app = Flask(__name__)

with open('config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)

data_path = config['path']
train_folder = config['train']
val_folder = config['val']

def get_images_list(folder_path, num_images=954):
    images = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    return images[:num_images]

def get_total_images(folder_path):
    images = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    return len(images)

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/training', methods=['GET', 'POST'])
def training():
    epochs = 0
    learning_rate = 0
    momentum = 0
    optimizer = " "
    resume = " "
    results = []

    train_images = get_images_list(os.path.join(data_path, train_folder))
    val_images = get_images_list(os.path.join(data_path, val_folder))
    total_train_images = get_total_images(os.path.join(data_path, train_folder))

    if request.method == "POST":
        yolo_model = YOLO("yolov8n.yaml")
        epochs = int(request.form['epochs'])
        learning_rate = float(request.form['learning_rate'])
        momentum = float(request.form['momentum'])
        optimizer = request.form['optimizer']
        resume = request.form['resume']

        results = yolo_model.train(data="config.yaml", epochs=epochs, lr0=learning_rate, momentum=momentum, optimizer=optimizer, resume=resume)
        return render_template('training.html', results=results)
    
    return render_template('training.html', train_images=train_images, val_images=val_images, total_train_images=total_train_images)

@app.route('/training/hasil')
def training_hasil():
    f1 = url_for('static', filename='training/F1_curve.png')
    precision = url_for('static', filename='training/P_curve.png')
    recall = url_for('static', filename='training/R_curve.png')
    confusion_matrix = url_for('static', filename='training/confusion_matrix_normalized.png')
    results = url_for('static', filename='training/results.png')

    return render_template('training_hasil.html', confusion_matrix=confusion_matrix,
                           results=results, f1=f1, precision=precision, recall=recall)

@app.route('/testing', methods=['GET', 'POST'])
def testing():
    final_string = ''
    if request.method == "POST":
        results = {}
        mot_tracker = Sort()

         # load model - vehicle & plate
        coco_model = YOLO('yolov8n.pt')
        license_plate_detector = YOLO('LPR/models/best.pt')

        # load model - character
        json_file = open('models/CRNN.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights("LPR/models/character_recognition_weight.h5")
        labels = LabelEncoder()
        labels.classes_ = np.load('LPR/models/character_classes.npy')

        vehicles = [2, 3, 5, 7]

        if 'input-image' in request.files:
            file = request.files['input-image']

            image = cv2.imread(file)

            detections = coco_model(image)[0]
            detections_ = []
            for detection in detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = detection
                if int(class_id) in vehicles:
                    detections_.append([x1, y1, x2, y2, score])

            track_ids = mot_tracker.update(np.asarray(detections_))

            class_names = coco_model.names
            vehicle_class_name = class_names[int(class_id)]

            def sort_contours(cnts,reverse = False):
                i = 0
                boundingBoxes = [cv2.boundingRect(c) for c in cnts]
                (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                                    key=lambda b: b[1][i], reverse=reverse))
                return cnts

            def predict_from_model(image,model,labels):
                image = cv2.resize(image,(80,80))
                image = np.stack((image,)*3, axis=-1)
                prediction = labels.inverse_transform([np.argmax(model.predict(image[np.newaxis,:]))])
                return prediction
            
            results = {}
            for license_plate in license_plate_detector(image)[0].boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = license_plate
                xvehicle1, yvehicle1, xvehicle2, yvehicle2, vehicle_id = get_vehicle(license_plate, track_ids)

                if (len(image)):
                    license_plate_crop = image[int(y1):int(y2), int(x1): int(x2), :]

                    license_plate_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                    license_plate_blur = cv2.GaussianBlur(license_plate_gray, (7, 7), 0)
                    
                    license_plate_binary = cv2.threshold(license_plate_blur, 220, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
                    license_plate_kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                    license_plate = cv2.morphologyEx(license_plate_binary, cv2.MORPH_DILATE, license_plate_kernel3)

                    license_plate_text, license_plate_text_score = read_license_plate(license_plate)

                    cont, _  = cv2.findContours(license_plate_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    test_roi = license_plate_crop.copy()

                    crop_characters = []

                    if vehicle_class_name.lower() in ['car', 'cars', 'bus', 'truck']:
                        digit_w, digit_h = 30, 60 # Mobil
                        for c in sort_contours(cont):
                            (x, y, w, h) = cv2.boundingRect(c)
                            ratio = h/w
                            if 1<=ratio<=3.5:
                                if h/license_plate_crop.shape[0]>=0.35:
                                    cv2.rectangle(test_roi, (x, y), (x + w, y + h), (0, 255,0), 3)
                                    curr_num = license_plate[y:y + h, x:x + w]
                                    curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                                    _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                                    crop_characters.append(curr_num)

                    elif vehicle_class_name.lower() in ['motorcycle', 'motorcycles']:
                        digit_w, digit_h = 10, 20 # Motor
                        for c in sort_contours(cont):
                            (x, y, w, h) = cv2.boundingRect(c)
                            ratio = h/w
                            if 1<=ratio<=3.5:
                                if h/license_plate_crop.shape[0]>=0.35:
                                    cv2.rectangle(test_roi, (x, y), (x + w, y + h), (0, 255,0), 3)
                                    curr_num = license_plate[y:y + h, x:x + w]
                                    curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                                    _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                                    crop_characters.append(curr_num)

                    if len(crop_characters) > 0:
                        print("Detect {} letters...".format(len(crop_characters)))
                        fig = plt.figure(figsize=(4, 2))
                        plt.axis(False)
                        plt.imshow(test_roi)
                        plt.savefig("LPR/static/testing/character.png",dpi=300)

                        fig = plt.figure(figsize=(15,3))
                        cols = len(crop_characters)
                        grid = gridspec.GridSpec(ncols=cols,nrows=1,figure=fig)
                        final_string = ''
                        for i,character in enumerate(crop_characters):
                            fig.add_subplot(grid[i])
                            title = np.array2string(predict_from_model(character,model,labels))
                            plt.title('{}'.format(title.strip("'[]"),fontsize=20))
                            final_string+=title.strip("'[]")
                            plt.axis(False)
                            plt.imshow(character,cmap='gray')

                            print(final_string)
                            plt.savefig('LPR/static/testing/deteksi_karakter.png', dpi=300)
                        
                        # save hasil segmentasi karakter 
                        fig = plt.figure(figsize=(14,4))
                        grid = gridspec.GridSpec(ncols=len(crop_characters),nrows=1,figure=fig)
                        for i in range(len(crop_characters)):
                            fig.add_subplot(grid[i])
                            plt.axis(False)
                            plt.imshow(crop_characters[i],cmap="gray")
                        plt.savefig("LPR/static/testing/segmentasi_karakter.png",dpi=300)
                        
                    else:
                        print("No letters detected.")
                        final_string = 'Tidak Terdeteksi'
                        fig = plt.figure(figsize=(4, 2))
                        plt.axis(False)
                        plt.savefig("LPR/static/testing/character.png",dpi=300)
                        plt.savefig('LPR/static/testing/deteksi_karakter.png', dpi=300)
                        plt.savefig("LPR/static/testing/segmentasi_karakter.png",dpi=300)


                    if license_plate_text is not None:
                        results[vehicle_id] = {'vehicle': {'bbox': [xvehicle1, yvehicle1, xvehicle2, yvehicle2]},
                                        'license_plate': {'bbox': [x1, y1, x2, y2],
                                                            'text': license_plate_text,
                                                            'bbox_score': score,
                                                            'text_score': license_plate_text_score}}
                        # bounding box plat
                        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 15)  # Kotak Merah

                    elif license_plate_text is None:
                        # bounding box plat
                        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 15)  # Kotak Merah

                        # label plat 
                        cv2.putText(image, f'Tidak terdeteksi', (int(x1), int(y1) - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 5) # Teks Merah
                    
                    # save hasil preprocessing 
                    fig = plt.figure(figsize=(12,7))
                    plt.rcParams.update({"font.size":18})
                    grid = gridspec.GridSpec(ncols=2,nrows=3,figure = fig)
                    plot_image = [license_plate_crop, license_plate_gray, license_plate_blur, license_plate_binary, license_plate]
                    plot_name = ["plate","grayscaling","bluring","binarization","dilation"]
                    for i in range(len(plot_image)):
                        fig.add_subplot(grid[i])
                        plt.axis(False)
                        plt.title(plot_name[i])
                        if i ==0:
                            plt.imshow(plot_image[i])
                        else:
                            plt.imshow(plot_image[i],cmap="gray")
                    plt.savefig("LPR/static/testing/preprocessing.png", dpi=300)

            output_image_path = 'LPR/static/testing/output_image.jpg'
            cv2.imwrite(output_image_path, image)

            return redirect(url_for('testing_hasil',
                                    preprocessing_path="LPR/static/testing/preprocessing.png",
                                    segmentasi_karakter_path="LPR/static/testing/segmentasi_karakter.png",
                                    character_path="LPR/static/testing/character.png",
                                    deteksi_karakter_path = "LPR/static/testing/deteksi_karakter.png",
                                    image_path=output_image_path, final_string=final_string))

        elif 'input-folder' in request.files:
            folder = request.files.getlist("input-folder")

            folder = 'LPR/static/testing pic/K3'
            results_filename = ""
            for filename in os.listdir(folder):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(folder, filename)

                    # Load image
                    image = cv2.imread(image_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    # load model - vehicle & plate
                    results = {}
                    mot_tracker = Sort()
                    coco_model = YOLO('yolov8n.pt')
                    license_plate_detector = YOLO('LPR/models/best.pt')

                    # load model - character
                    json_file = open('models/CRNN.json', 'r')
                    loaded_model_json = json_file.read()
                    json_file.close()
                    model = model_from_json(loaded_model_json)
                    model.load_weights("LPR/models/character_recognition_weight.h5")
                    labels = LabelEncoder()
                    labels.classes_ = np.load('LPR/models/character_classes.npy')

                    vehicles = [2, 3, 5, 7]

                    detections = coco_model(image)[0]
                    detections_ = []
                    for detection in detections.boxes.data.tolist():
                        x1, y1, x2, y2, score, class_id = detection
                        if int(class_id) in vehicles:
                            detections_.append([x1, y1, x2, y2, score])

                    track_ids = mot_tracker.update(np.asarray(detections_))

                    class_names = coco_model.names
                    vehicle_class_name = class_names[int(class_id)]

                    def sort_contours(cnts,reverse = False):
                        i = 0
                        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
                        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                                            key=lambda b: b[1][i], reverse=reverse))
                        return cnts

                    def predict_from_model(image,model,labels):
                        image = cv2.resize(image,(80,80))
                        image = np.stack((image,)*3, axis=-1)
                        prediction = labels.inverse_transform([np.argmax(model.predict(image[np.newaxis,:]))])
                        return prediction
                    
                    results_filename = os.path.join(folder, 'results.txt')
                    with open(results_filename, 'a') as results_file:
                        results_file.write(f"{filename}, ")
                        for license_plate in license_plate_detector(image)[0].boxes.data.tolist():
                            x1, y1, x2, y2, score, class_id = license_plate
                            xvehicle1, yvehicle1, xvehicle2, yvehicle2, vehicle_id = get_vehicle(license_plate, track_ids)

                            if (len(image)):
                                license_plate_crop = image[int(y1):int(y2), int(x1): int(x2), :]

                                license_plate_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                                license_plate_blur = cv2.GaussianBlur(license_plate_gray, (7, 7), 0)
                                license_plate_binary = cv2.threshold(license_plate_blur, 220, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] #
                                license_plate_kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                                license_plate = cv2.morphologyEx(license_plate_binary, cv2.MORPH_DILATE, license_plate_kernel3)

                                license_plate_text, license_plate_text_score = read_license_plate(license_plate)

                                cont, _  = cv2.findContours(license_plate_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                                test_roi = license_plate_crop.copy()

                                crop_characters = []

                                if vehicle_class_name.lower() in ['car', 'cars', 'bus', 'truck']:
                                    digit_w, digit_h = 30, 60 # Mobil
                                    for c in sort_contours(cont):
                                        (x, y, w, h) = cv2.boundingRect(c)
                                        ratio = h/w
                                        if 1<=ratio<=3.5:
                                            if h/license_plate_crop.shape[0]>=0.35:
                                                cv2.rectangle(test_roi, (x, y), (x + w, y + h), (0, 255,0), 3)
                                                curr_num = license_plate[y:y + h, x:x + w]
                                                curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                                                _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                                                crop_characters.append(curr_num)

                                elif vehicle_class_name.lower() in ['motorcycle', 'motorcycles']:
                                    digit_w, digit_h = 10, 20 # Motor
                                    for c in sort_contours(cont):
                                        (x, y, w, h) = cv2.boundingRect(c)
                                        ratio = h/w
                                        if 1<=ratio<=3.5:
                                            if h/license_plate_crop.shape[0]>=0.35:
                                                cv2.rectangle(test_roi, (x, y), (x + w, y + h), (0, 255,0), 3)
                                                curr_num = license_plate[y:y + h, x:x + w]
                                                curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                                                _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                                                crop_characters.append(curr_num)

                                if len(crop_characters) > 0:
                                    print("Detect {} letters...".format(len(crop_characters)))

                                    final_string = ''
                                    for i,character in enumerate(crop_characters):
                                        title = np.array2string(predict_from_model(character,model,labels))
                                        final_string+=title.strip("'[]")
                                        print(final_string)
                                    
                                else:
                                    print("No letters detected.")
                                    final_string = ''

                                if license_plate_text is not None:
                                    results[vehicle_id] = {'vehicle': {'bbox': [xvehicle1, yvehicle1, xvehicle2, yvehicle2]},
                                                    'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                        'text': license_plate_text,
                                                                        'bbox_score': score,
                                                                        'text_score': license_plate_text_score}}
                                    # bounding box plat
                                    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 15)  # Kotak Merah

                                elif license_plate_text is None:
                                    # bounding box plat
                                    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 15)  # Kotak Merah

                                    # label plat 
                                    cv2.putText(image, f'Tidak terdeteksi', (int(x1), int(y1) - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 5) # Teks Merah
                                
                                results_file.write(f"{final_string}, ")

                                filename = os.path.splitext(filename)[0]

                                prefixes_remove = ["K1_", "K2_", "K3_", "K4_", "K5_"]
                                for prefix in prefixes_remove:
                                    if filename.startswith(prefix):
                                        filename = filename[len(prefix):]

                                true_characters = "".join(char for char in filename if char.isalnum())
                                correct_predictions = sum([1 for pred, true_char in zip(final_string, true_characters) if pred == true_char])
                                total_characters = len(true_characters)

                                accuracy = correct_predictions / total_characters
                                results_file.write(f"{accuracy:.2%}")

                            results_file.write("\n")  

            return redirect(url_for('testing_hasil', results_path=results_filename))

    return render_template('testing.html')

@app.route('/testing/hasil')
def testing_hasil():
    output_image = url_for('static', filename='testing/output_image.jpg')
    preprocessing = url_for('static', filename='testing/preprocessing.png')
    deteksi_karakter = url_for('static', filename='testing/deteksi_karakter.png')
    character = url_for('static', filename='testing/character.png')
    final_string = request.args.get('final_string')
    segmentasi_karakter = url_for('static', filename='testing/segmentasi_karakter.png')

    return render_template('testing_hasil.html', preprocessing=preprocessing,
                           segmentasi_karakter=segmentasi_karakter,
                           character=character,
                           deteksi_karakter=deteksi_karakter,
                           output_image=output_image, final_string=final_string)

# CRUD
upload_folder = 'LPR/static/dataset'
app.config['UPLOAD_FOLDER'] = upload_folder

def ensure_upload_folder():
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/dataset')
def dataset():
    ensure_upload_folder()
    images = os.listdir(app.config['UPLOAD_FOLDER'])
    return render_template('dataset.html', images=images)

@app.route('/add', methods=['POST'])
def add():
    if request.method == 'POST':
        ensure_upload_folder()
        image = request.files['input-image']
        if image:
            filename = secure_filename(image.filename)
            image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return redirect(url_for('dataset'))

@app.route('/edit/<filename>', methods=['GET', 'POST'])
def edit(filename):
    ensure_upload_folder()
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if request.method == 'POST':
        new_filename = secure_filename(request.form['new_filename'])
        new_image = request.files['new_image']

        if new_image.filename != '':
            new_image.save(os.path.join(app.config['UPLOAD_FOLDER'], new_filename))

        os.rename(filepath, os.path.join(app.config['UPLOAD_FOLDER'], new_filename))
        return redirect(url_for('dataset'))
    
    return render_template('edit.html', filename=filename)

@app.route('/delete/<filename>')
def delete(filename):
    ensure_upload_folder()
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(filepath):
        os.remove(filepath)
    return redirect(url_for('dataset'))

if __name__ == '__main__':
    app.run(debug=True)
