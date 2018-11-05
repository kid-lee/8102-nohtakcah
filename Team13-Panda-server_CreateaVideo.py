#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 03:18:17 2018

@author: jaime
"""
import cv2
import numpy as np

import imageio
import face_recognition

import timeit
import csv

start = 0
end = 0
# Get a reference to webcam #0 (the default one)
#video_capture = cv2.VideoCapture("image/video3.MOV")

# Load a sample picture and learn how to recognize it.
tom_image = face_recognition.load_image_file("image/tom.jpg")
tom_face_encoding = face_recognition.face_encodings(tom_image)[0]

# Load a second sample picture and learn how to recognize it.
sindy_image = face_recognition.load_image_file("image/sindy.jpg")
sindy_face_encoding = face_recognition.face_encodings(sindy_image)[0]

oska_image = face_recognition.load_image_file("image/oska.jpg")
oska_face_encoding = face_recognition.face_encodings(oska_image)[0]

jaime_image = face_recognition.load_image_file("image/jaime.jpg")
jaime_face_encoding = face_recognition.face_encodings(jaime_image)[0]

harrison_image = face_recognition.load_image_file("image/Harrisonford.jpg")
harrison_face_encoding = face_recognition.face_encodings(harrison_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    tom_face_encoding,
    sindy_face_encoding,
    oska_face_encoding,
    jaime_face_encoding,
    harrison_face_encoding
]
known_face_names = [
    "tom",
    "Sindy chiu",
    "oska",
    "jaime",
    "harrison"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []

recognitions = []

def FaceRecognition(Image):
    flag  = 0
    # Resize frame of video to 1/4 size for faster face recognition processing
    process_this_frame = True
    frame = Image
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                flag = 1
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    #cv2.imshow('Face_Recognition', frame)
    return flag

def resizeimage(Image,w,h):
    RImage = cv2.resize(Image,(int(w),int(h)))
    return RImage


def playVideo(video_file,resizeby):
    video = imageio.get_reader(video_file,'ffmpeg')
    #video = imageio.get_reader(video_file)
    metadata = video.get_meta_data()
    print(metadata)
    W = metadata['size'][0]
    H = metadata['size'][1]
    w = W/resizeby
    h = H/resizeby
    for frame in video:
        fr = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        cv2.imshow('frame',fr)
        Resizedfr = resizeimage(fr,w,h)
        cv2.imshow('Rframe',Resizedfr)
        if cv2.waitKey(40) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    
def playVideoSampledbytime(video_file,resizeby,seconds):
    video = imageio.get_reader(video_file,'ffmpeg')
    #video = imageio.get_reader(video_file)
    metadata = video.get_meta_data()
    print(metadata)
    W = metadata['size'][0]
    H = metadata['size'][1]
    w = W/resizeby
    h = H/resizeby
    
    fps = int(metadata['fps'])
    print(fps)
    sample = int(fps*seconds)
    print(sample)
    totalframes = int(metadata['nframes'])
    if sample == 0:
        sample = 1

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('HarriVideo_V02.avi', fourcc, int(metadata['fps']), (w, h))

    for i in range(0, (totalframes-100),sample):
        frame = video.get_data(i)
        fr = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        #cv2.imshow('Original Video',fr)
        Resizedfr = resizeimage(fr,w,h)
        #cv2.imshow('Processed Video', Resizedfr)
        #Insert face_recognitioncode
        Tom = 0
        #recognitions.append(FaceRecognition(Resizedfr))
        Tom = FaceRecognition(Resizedfr)
        recognitions.append(Tom)
        #np.savetxt("Results.txt", recognitions)

        if cv2.waitKey(40) & 0xFF == ord('q'):
            break

    end = timeit.timeit()
    print("Time elapsed = ", end - start)
    #np.savetxt("Results.txt",int(recognitions))
    print("Saving CSVFile on: ResultsSampligbySec_harris_V02.csv")
    field = ['Second','Appears_in_the_Movie']
    #np.savetxt("ResultsSampligbySec_harri.csv", recognitions, delimiter=',')
    with open(r'ResultsSampligbySec_harri_V02.csv','a') as f:
        csv.writer(f).writerow(field)
    print("File Saved Correctly.")
    # Release everything if job is finished
    print("Saving Video in HarriVideo.avi")
    # *************
    # WriteVideo
    print(len(recognitions))
    for i in range(len(recognitions)):
        with open(r'ResultsSampligbySec_harri.csv', 'a') as f:
            print(i)
            csv.writer(f).writerow([i, recognitions[i]])
        if int(recognitions[i]) == 1:
            print("Tom")
            frame = video.get_data((i*fps))
            fr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out.write(fr)
    # *************


    out.release()
    print("Saving Video Done")
    cv2.destroyAllWindows()
    print("Process Finished")
 
filename = 'image/video2.mp4'
#playVideo(filename,2)
ReduceBy = 1 #
SampleBy = 1# Second 17：2：50
start = timeit.timeit()
playVideoSampledbytime(filename,ReduceBy,SampleBy)
#45