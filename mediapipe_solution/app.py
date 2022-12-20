import cv2
import numpy as np
import mediapipe as mp
from gui import GUI
import tensorflow as tf
from tkinter import *
#python image library to process images
from PIL import ImageTk, Image
from spellchecker import SpellChecker
import time
from tensorflow.keras.models import load_model

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def pre_process_landmark(landmark_list):

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        # take the difference between 
        # each coordinates with the base value at zero
        landmark_list[index][0] = np.abs(landmark_list[index][0] - base_x)
        landmark_list[index][1] = np.abs(landmark_list[index][1] - base_y)

    flattened = []
    # flatten it from Nx2 to 
    # Mx1 by pushing the y values to the same
    # column as x
    for i in range(len(landmark_list)):
        flattened.append(landmark_list[i][0])
        flattened.append(landmark_list[i][1])


    normalized = []
    # Normalization
    for i in range(len(flattened)):
        # normalize the coordinate values so it is 
        # between [0,1] relative to the biggest value in the flatten array
        normalized.append(flattened[i]/max(flattened))

    return normalized

def landmark_list(image, landmarks):
    height, width, _ = image.shape

    landmark_point = []

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * width), width - 1)
        landmark_y = min(int(landmark.y * height), height - 1)

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def prediction_to_str(pred):
    if pred+66 == ord(']'):
        return 'space'
    elif pred+65 == ord('['):
        return 'del'
    else:
        return chr(pred + 65)

def update_signed(pred, signed):
    if pred == 'del':
        signed = signed[:-1]
    elif pred == 'space':
        signed += ' '
    else:
        signed += pred
    
    return signed

def start_gui(title, size):
    gui = GUI(title, size)
    print("Started GUI")
    gui_frame = gui.create_frame(500, 500, 'center', 1, 0, 'green')
    vid_label = Label(gui_frame)
    vid_label.grid()
    
    return gui, vid_label

def update_frame(image, vid_label):
    image_fromarray = Image.fromarray(image)
    imgtk = ImageTk.PhotoImage(image=image_fromarray)
    
    #updates the image in the video frame
    vid_label.imgtk = imgtk
    vid_label.config(image=imgtk)
    
def auto_correct(word): 
    myChecker = SpellChecker()
    correct_word = myChecker.correction(word)
    if correct_word == None:
        return "error"
    
    return correct_word

def add_charToWord(word, char):
    #because string are passed by reference 
    # here we need to make another variable temp
    temp_word = word
    if char == 'del':
        temp_word = temp_word[:-1]
        print("character is deleted: ", char.lower())
    elif char == 'space':
        temp_word = ""
    else:
        temp_word += char.lower()
        print("character has been added: ", char.lower())
        
    return [temp_word, char]
    
    
    
    
def getTextbox(gui):
    curr_char = None 
    prev_char = None
    word = ""
    sentence = ""
    
    labels_num = 5
    labels = ['current char', 'probability', 'original word', 'corrected word', 'sentence']
    
    #create labels for the gui
    Labels, entryboxes = gui.create_labels(labels_num, labels, 'nw', 0, 0, y_spacing=0.06, create_entrybox_per_label=1)
    
    #resizing the entry box to git the screen
    entryboxes['corrected word_entrybox'].config(width=18)
    entryboxes['original word_entrybox'].config(width=18)
    entryboxes['sentence_entrybox'].config(width=18, height=18)
    
    cc_entrybox = entryboxes['current char_entrybox']

    pb_entrybox = entryboxes['probability_entrybox']

    ow_entrybox = entryboxes['original word_entrybox']


    cw_entrybox = entryboxes['corrected word_entrybox']


    sent_entrybox = entryboxes['sentence_entrybox']

    names = ['vid_label', 'hands', 'cc_box', 'pb_box', 'ow_box', 'cw_box', 'sent_box']

    return cc_entrybox, pb_entrybox, ow_entrybox, cw_entrybox, sent_entrybox, names

def frame_video_stream(names, threshold, curr_char, prev_char, repeat, word, sentence, *args):
    kwargs = dict(zip(names, args))
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
    # If loading a video, use 'break' instead of 'continue'.
        # continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    
    # convert image into usable color space BGR -> RBG
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #process image
    results = kwargs['hands'].process(image)
    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #get the prediction from the hand images
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            #process landmarks so it scales with the size of the image
            # by multiplying it with width and height of the images
            # removing the z-index
            landmarks = landmark_list(image, hand_landmarks)
            
            #flatten + normaliee the landmark array 
            # into one array and process it
            processed = pre_process_landmark(landmarks)
            # pass processed array into model for predicttion 
            # prediction is stored as string
            # print("model: ", model.predict(np.array([processed])))
            # print("processed: ", processed)
            # get a prediction array wthi 26 
            prediction = prediction_to_str(np.argmax(np.squeeze(model.predict(np.array([processed])))))
            char_prob = '{0:.2f}'.format(np.amax(np.squeeze(model.predict(np.array([processed])))))
        
            # draw the landmarks on the hand in the video
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
            
            #update the gui frame label 
            image_copy = cv2.resize(image, (int(image.shape[1]*0.5), int(image.shape[0]*0.5)), interpolation = cv2.INTER_AREA)
            image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
            update_frame(image_copy, kwargs['vid_label'])
            #clear the entry boxes on each run
            kwargs['cc_box'].delete('1.0', 'end')
            kwargs['pb_box'].delete('1.0', 'end')
            kwargs['pb_box'].insert('end', char_prob)
            kwargs['cc_box'].insert('end', prediction)
            
            # update the threshold
            # figure out a way to spell two consecutive letters
            curr_char = prediction
            print("prediction: ", prediction)
            # only reset repeat count
            # if current character is different from the previous one
            if curr_char == prev_char and curr_char != 'del':
                repeat +=1
            # only reset if the probiliaty of letter is higher thank threshold
            # if not then signers will reset the repeat count
            # with the slightest change
            elif(float(char_prob) > threshold):
                repeat = 0
                
            # if the character is not repeated more than once
            # AND the probability is more threshold
            if (repeat < 1) :
                #the below print statement is related to the formatter
                #print(pred)
                #add character to curent word
                temp = add_charToWord(word, curr_char)
                threshold = 0.95
                if curr_char == 'del':
                    kwargs['ow_box'].delete('1.0', 'end')
                    kwargs['ow_box'].insert('end', temp[0].upper())
                elif curr_char != 'space' :
                    kwargs['ow_box'].insert('end', temp[1].upper())
                
                if (temp[0] == "") and (temp[1] != "del"):
                    sentence += auto_correct(word) + " "
                    kwargs['sent_box'].insert('end', auto_correct(word) + " ")
                    kwargs['ow_box'].delete('1.0', 'end')
                    kwargs['cw_box'].delete('1.0', 'end')
                    kwargs['cw_box'].insert('end', auto_correct(word))
                word = temp[0]

                prev_char = curr_char
            # decrease threshold if the same letter reminds at a low threshold for a long time
            else:
                threshold = threshold - 0.02
                
    # //////////////////
    # Flip the image horizontally for a selfie-view display.
        #put prediction at the top left hand corner
        # cv2.putText(image, 'pred: '+prediction, (12,48), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), thickness=1)
        # cv2.putText(image, signed, (12, 96), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), thickness=1)
        
        # #display the camera feed
        # cv2.imshow('MediaPipe Hands', image)
        # if cv2.waitKey(33) == ord(' '):
        #     signed = update_signed(prediction, signed)
        
    #refresh rate of the video frame
    kwargs['vid_label'].after(1, frame_video_stream, names, threshold, curr_char, prev_char, repeat, word, sentence, *args)

def main():
    model_save_path = './mediapipe_solution/keypoint_classifier2.hdf5'
    global model
    model = tf.keras.models.load_model(model_save_path)
    signed = ''
    prediction = ''
    threshold = float(0.95)
    
    curr_char = None
    prev_char = None
    repeat = 0
    word = ""
    sentence = ""

    title = "ASL Interpreter GUI"
    size = "800x800"

    gui, vid_label = start_gui(title, size)

    # For webcam input:
    global cap
    cap = cv2.VideoCapture(0)
    # success, image = cap.read()
    # issue 1 : scaling the image before passing it into update_frame
    # image = cv2.imread('./mediapipe_solution/img1.jpg')
    # print(image.shape)
    # image = cv2.resize(image, (int(image.shape[1]*0.3), int(image.shape[0]*0.3)), interpolation = cv2.INTER_AREA)
    # print(image.shape)
    # update_frame(image, vid_label)
    # if not success:
    #     print("Ignoring empty camera frame.")
    #Initiate pipeline for camera
    cc_entrybox, pb_entrybox, ow_entrybox, cw_entrybox, sent_entrybox, names = getTextbox(gui)
    print('cc_entrybox: ', cc_entrybox)
    image = cv2.imread('./mediapipe_solution/img1.jpg')
    # get a list of hands 
    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        # while cap.isOpened():
        # read from camera and get frames by frame
        # //////////////////
        frame_video_stream(names, threshold, curr_char, prev_char, repeat, word, sentence, vid_label,
                               hands, cc_entrybox, pb_entrybox, ow_entrybox, cw_entrybox, sent_entrybox)
        gui.root.mainloop()
        

    #closes camera after
    # cap.release()

  
if __name__ == '__main__':
  main()