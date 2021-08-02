# -*- coding:utf-8 -*-

import PySimpleGUI as sg
from PySimpleGUI.PySimpleGUI import RELIEF_SUNKEN
import cv2
import numpy as np



def inverse_color(img):
    img = (255 - img)
    return img


def preprocess(img, inverse = False, alpha=2.5, blur=5, threshold=37, adjustment=11, 
                dilate=3, iter_dilate=2,
                erode=3, iter_erode=2):
    debug_images = []

    debug_images.append(('original', img))

    # exposure
    alpha = float(alpha)
    exposured = cv2.multiply(img, np.array([alpha]))
    debug_images.append(('exposured', exposured))

    # gray
    gray = cv2.cvtColor(exposured, cv2.COLOR_BGR2GRAY)
    debug_images.append(('gray',gray))

    # blur
    blurred = cv2.GaussianBlur(gray, (blur, blur), 0)
    if inverse:
        blurred = inverse_color(blurred)
    debug_images.append(('blurred', blurred))

    # binary
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, threshold, adjustment)
    binary = inverse_color(binary)
    debug_images.append(('binary', binary))

    # dilate
    dilated = cv2.dilate(binary, (dilate, dilate), iterations=iter_dilate)
    debug_images.append(('dilated', dilated))

    # erode
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (erode, erode))
    eroded = cv2.erode(dilated, kernel, iterations=iter_erode)
    eroded = cv2.morphologyEx(eroded, cv2.MORPH_CLOSE, kernel) # close hole
    debug_images.append(('eroded', eroded))

    return(debug_images)



def sort_contours(cnts, method='left-to-right'):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == 'right-to-left' or method == 'bottom-to-top':
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == 'top-to-bottom' or method == 'bottom-to-top':
        i = 1
    
    # construct the list of bounding boxes and sort them from top to 
    # bottom
    bounding_boxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, bounding_boxes) = zip(*sorted(zip(cnts, bounding_boxes), 
                                            key=lambda b: b[1][i], reverse=reverse))
    
    # return the list of sorted contours and bounding boxes
    return cnts, bounding_boxes
    


def main():

    sg.theme('Black')

    frame_roi = sg.Frame(layout=[
        [sg.Text('Positon', size=(14,1)),
         sg.Slider((0,640), 320, 1, orientation='h', size=(30,15), key='-NONE SLIDER A-'),
         sg.Slider((0,480), 240, 1, orientation='h', size=(30,15), key='-NONE SLIDER B-')],
        [sg.Button('TopLeft', size=(10,1)),
         sg.Input(0, size=(30,15), key='-TOPLEFTX-'),
         sg.Input(0, size=(30,15), key=('-TOPLEFTY-'))],
        [sg.Button('TopRight', size=(10,1)),
         sg.Input(0, size=(30,15), key='-TOPRIGHTX-'),
         sg.Input(0, size=(30,15), key=('-TOPRIGHTY-'))],
        [sg.Button('BottomRight', size=(10,1)),
         sg.Input(0, size=(30,15), key='-BOTTOMRIGHTX-'),
         sg.Input(0, size=(30,15), key=('-BOTTOMRIGHTY-'))],
        [sg.Button('BottomLeft', size=(10,1)),
         sg.Input(0, size=(30,15), key='-BOTTOMLEFTX-'),
         sg.Input(0, size=(30,15), key=('-BOTTOMLEFTY-'))],
        [sg.Text('ROI size, W x H', size=(14,1)),
         sg.Slider((0,640), 200, 1, orientation='h', size=(30,15), key='-ROIWIDTH-'),
         sg.Slider((0,480), 100, 1, orientation='h', size=(30,15), key='-ROIHEIGHT-')],
        [sg.Text('', size=(14,1), justification='left')],
        [sg.Button('Clear', size=(10,2))]],
        title="ROI Setting",
        title_color='White',
        relief=sg.RELIEF_SUNKEN)

    frame_preprocess = sg.Frame(layout=[
        [sg.Frame(layout=[
            [sg.Text('Multiply', size=(14,1)),
             sg.Slider((0,10), 2, 0.1, orientation='h', size=(20,15), key='-EXPOSURE MULTIPLY-')]],
            title="Exposure")],
        [sg.Frame(layout=[
            [sg.Text('Kernel Size', size=(14,1)),
             sg.Slider((1,5), 2, 1, orientation='h', size=(20,15), key='-BLUR KERNEL-'),
             sg.Text('Inverse', size=(14,1)),
             sg.Checkbox("", size=(17,15), key='-CKB INVERSE-')]], 
            title="Blur")],
        [sg.Frame(layout=[
            [sg.Text('Threshold', size=(14,1)),
             sg.Slider((1,50), 20, 1, orientation='h', size=(20,15), key='-BINARY THRESH-'),
             sg.Text('Adjustment', size=(14,1)),
             sg.Slider((0,50), 11, 1, orientation='h', size=(20,15), key='-BINARY ADJUST-')]],
            title='Binary')],
        [sg.Frame(layout=[
            [sg.Text('Kernel Size', size=(14,1)),
             sg.Slider((1,5), 2, 1, orientation='h', size=(20,15), key='-DILATE KERNEL-'),
             sg.Text('Iterations', size=(14,1)),
             sg.Slider((0,10), 1, 1, orientation='h', size=(20,15), key='-DILATE ITER-')]],
            title='Dilate')],
        [sg.Frame(layout=[
            [sg.Text('Kernel Size', size=(14,1)),
             sg.Slider((1,5), 2, 1, orientation='h', size=(20,15), key='-ERODE KERNEL-'),
             sg.Text('Iterations', size=(14,1)),
             sg.Slider((0,10), 1, 1, orientation='h', size=(20,15), key='-ERODE ITER-')]],
            title='Erode')]
        ],
        title="Preprocessing",
        title_color="White",
        relief=sg.RELIEF_SUNKEN)

    layout = [
        [sg.Image(filename='', key='-IMAGE1-'),
         sg.Image(filename='', key='-IMAGE2-')],
        [frame_roi, frame_preprocess],       
        [sg.Text('', size=(14,1), justification='left')],
        [sg.Button('Exit', size=(10,2))]
    ]

    window = sg.Window('NUMBERS', layout, location=(200,200))

    cap = cv2.VideoCapture(0)

    while True:

        event, values = window.read(timeout=20)

        if event == 'Exit' or event == sg.WIN_CLOSED:
            break

        ret, frame = cap.read()

        temp = frame.copy()

        if event == 'TopLeft':
            window['-TOPLEFTX-'].update(int(values['-NONE SLIDER A-']))
            window['-TOPLEFTY-'].update(int(values['-NONE SLIDER B-']))
        if event == 'TopRight':
            window['-TOPRIGHTX-'].update(int(values['-NONE SLIDER A-']))
            window['-TOPRIGHTY-'].update(int(values['-NONE SLIDER B-']))
        if event == 'BottomRight':
            window['-BOTTOMRIGHTX-'].update(int(values['-NONE SLIDER A-']))
            window['-BOTTOMRIGHTY-'].update(int(values['-NONE SLIDER B-']))
        if event == 'BottomLeft':
            window['-BOTTOMLEFTX-'].update(int(values['-NONE SLIDER A-']))
            window['-BOTTOMLEFTY-'].update(int(values['-NONE SLIDER B-']))        
        if event == 'Clear':
            window['-TOPLEFTX-'].update(0)
            window['-TOPLEFTY-'].update(0)
            window['-TOPRIGHTX-'].update(0)
            window['-TOPRIGHTY-'].update(0)
            window['-BOTTOMRIGHTX-'].update(0)
            window['-BOTTOMRIGHTY-'].update(0)
            window['-BOTTOMLEFTX-'].update(0)
            window['-BOTTOMLEFTY-'].update(0)

        cx = int( values['-NONE SLIDER A-'] )
        cy = int( values['-NONE SLIDER B-'] )
        if cx != 0 or cy != 0:
            w, h = 640, 480                
            cv2.line(frame, (cx, 0), (cx, h - 1), (0,255,0))
            cv2.line(frame, (0, cy), (w - 1, cy), (0,255,0))

        if values['-TOPLEFTX-'] != '0' or values['-TOPLEFTY-'] != '0':
            cv2.circle(frame, (int(values['-TOPLEFTX-']), int(values['-TOPLEFTY-'])), 2, (0,0,255), 2)
        if values['-TOPRIGHTX-'] != '0' or values['-TOPRIGHTY-'] != '0':
            cv2.circle(frame, (int(values['-TOPRIGHTX-']), int(values['-TOPRIGHTY-'])), 2, (0,0,255), 2)
        if values['-TOPLEFTX-'] != '0' or values['-TOPLEFTY-'] != '0':
            cv2.circle(frame, (int(values['-BOTTOMRIGHTX-']), int(values['-BOTTOMRIGHTY-'])), 2, (0,0,255), 2)
        if values['-TOPLEFTX-'] != '0' or values['-TOPLEFTY-'] != '0':
            cv2.circle(frame, (int(values['-BOTTOMLEFTX-']), int(values['-BOTTOMLEFTY-'])), 2, (0,0,255), 2)

        if (values['-TOPLEFTX-'] != '0' and values['-TOPLEFTY-'] != '0' and
            values['-TOPRIGHTX-'] != '0' and values['-TOPRIGHTY-'] != '0' and
            values['-BOTTOMRIGHTX-'] != '0' and values['-BOTTOMRIGHTY-'] != '0' and
            values['-BOTTOMLEFTX-'] != '0' and values['-BOTTOMLEFTY-'] != '0'):
            cv2.putText(frame, "ROI fixed", (10,30), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2, cv2.LINE_AA)
            
            pnts = [(int(values['-TOPLEFTX-']), int(values['-TOPLEFTY-'])),
                    (int(values['-TOPRIGHTX-']), int(values['-TOPRIGHTY-'])),
                    (int(values['-BOTTOMRIGHTX-']), int(values['-BOTTOMRIGHTY-'])),
                    (int(values['-BOTTOMLEFTX-']), int(values['-BOTTOMLEFTY-'] ))]
            pnts = np.float32(pnts)

            width = int(values['-ROIWIDTH-'])
            height = int(values['-ROIHEIGHT-'])
            pnts2 = np.float32([[0, 0], [width,0], [width, height], [0, height]])
            
            M = cv2.getPerspectiveTransform(pnts, pnts2)
            roi = cv2.warpPerspective(temp, M, (width, height))

            roi = preprocess(roi, 
                inverse = values['-CKB INVERSE-'],
                alpha = values['-EXPOSURE MULTIPLY-'],
                blur = int(values['-BLUR KERNEL-']) * 2 + 1,
                threshold= int(values['-BINARY THRESH-']) * 2 + 1,
                adjustment = int(values['-BINARY ADJUST-']),
                dilate = int(values['-DILATE KERNEL-']) * 2 + 1,
                iter_dilate = int(values['-DILATE ITER-']),
                erode = int(values['-ERODE KERNEL-']) * 2 + 1,
                iter_erode = int(values['-ERODE ITER-']))

            
            imgbytes2 = cv2.imencode('.png', roi[6][1])[1].tobytes()
            window['-IMAGE2-'].update(data=imgbytes2)

        

        imgbytes = cv2.imencode('.png', frame)[1].tobytes()
        window['-IMAGE1-'].update(data=imgbytes)

    window.close()


main()

