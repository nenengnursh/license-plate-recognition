import string
import easyocr

reader = easyocr.Reader(['en'], gpu=True)

dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5',
                    'B' : '8',
                    'L' : '4'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S',
                    '8' : 'B',
                    '4' : 'L'}

dict_char_to_char = { 'C' : 'G',
                      'O' : 'D',
                      'B' : 'E',
                      'E' : 'B'}

def license_complies_format(text):
    # 9 Nomor
    if len(text) >= 9:
       if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
        (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and \
        (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2] in dict_char_to_int.keys()) and \
        (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in dict_char_to_int.keys()) and \
        (text[4] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[4] in dict_char_to_int.keys()) and \
        (text[5] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[5] in dict_char_to_int.keys()) and \
        (text[6] in string.ascii_uppercase or text[6] in dict_int_to_char.keys()) and \
        (text[7] in string.ascii_uppercase or text[7] in dict_int_to_char.keys()) and \
        (text[8] in string.ascii_uppercase or text [8] in dict_char_to_char.keys()):
           return True
    
    # 8 Nomor
    elif len(text) >= 8:
        if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
           (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and \
           (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2] in dict_char_to_int.keys()) and \
           (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in dict_char_to_int.keys()) and \
           (text[4] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[4] in dict_char_to_int.keys()) and \
           (text[5] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[5] in dict_char_to_int.keys()) and \
           (text[6] in string.ascii_uppercase or text[6] in dict_int_to_char.keys()) and \
           (text[7] in string.ascii_uppercase or text[7] in dict_int_to_char.keys()):
            return True
    
    # 8 Nomor
    elif len(text) >= 8:
        if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
           (text[1] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[1] in dict_char_to_int.keys()) and \
           (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2] in dict_char_to_int.keys()) and \
           (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in dict_char_to_int.keys()) and \
           (text[4] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[4] in dict_char_to_int.keys()) and \
           (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()) and \
           (text[6] in string.ascii_uppercase or text[6] in dict_int_to_char.keys()) and \
           (text[7] in string.ascii_uppercase or text[7] in dict_int_to_char.keys()):
            return True
    
    # 7 Nomor
    elif len(text) >= 7:
        if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
          (text[1] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[1] in dict_char_to_int.keys()) and \
          (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2] in dict_char_to_int.keys()) and \
          (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in dict_char_to_int.keys()) and \
          (text[4] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[4] in dict_char_to_int.keys()) and \
          (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()) and \
          (text[6] in string.ascii_uppercase or text[6] in dict_int_to_char.keys()):
           return True
    
    # 6 Nomor
    elif len(text) >= 6:
        if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
           (text[1] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[1] in dict_char_to_int.keys()) and \
           (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2] in dict_char_to_int.keys()) and \
           (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in dict_char_to_int.keys()) and \
           (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and \
           (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()):
            return True
    
    # 5 Nomor
    elif len(text) >= 5:
        if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
           (text[1] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[1] in dict_char_to_int.keys()) and \
           (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2] in dict_char_to_int.keys()) and \
           (text[3] in string.ascii_uppercase or text[3] in dict_int_to_char.keys()) and \
           (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()):
            return True
        
    # 5 Nomor
    elif len(text) >= 5:
        if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
           (text[1] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[1] in dict_char_to_int.keys()) and \
           (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2] in dict_char_to_int.keys()) and \
           (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in dict_char_to_int.keys()) and \
           (text[4] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[4] in dict_char_to_int.keys()):
            return True
    
    else:
        return False


def format_license(text):
    license_plate_ = ''
    mapping = {0: dict_int_to_char, 1: dict_int_to_char, 1: dict_char_to_int, 2: dict_char_to_int, 3: dict_char_to_int,
               4: dict_char_to_int, 5: dict_char_to_int, 6: dict_int_to_char, 7: dict_int_to_char,
               8: dict_int_to_char, 8: dict_char_to_char}
    
    for j in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
        if j < len(text) and text[j] in mapping[j].keys():
            license_plate_ += mapping[j][text[j]]
        else:
            license_plate_ += text[j] if j < len(text) else ''

    return license_plate_


def read_license_plate(license_plate_crop):
    detections = reader.readtext(license_plate_crop)

    for detection in detections:
        bbox, text, score = detection

        text = text.upper().replace(' ', '')
        
        # Mengganti semua simbol dengan string kosong
        translator = str.maketrans('', '', string.punctuation)
        text = text.translate(translator)

        if license_complies_format(text):
            return format_license(text), score
        else:
            return f"{text}", score

    return None, None


def get_vehicle(license_plate, vehicle_track_ids):
    x1, y1, x2, y2, score, class_id = license_plate

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xvehicle1, yvehicle1, xvehicle2, yvehicle2, vehicle_id = vehicle_track_ids[j]

        if x1 > xvehicle1 and y1 > yvehicle1 and x2 < xvehicle2 and y2 < yvehicle2:
            vehicle_indx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[vehicle_indx]

    return -1, -1, -1, -1, -1
