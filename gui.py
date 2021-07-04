from tkinter import *
from PIL import ImageTk, Image


root= Tk()
root.title("Facial Recognition System")


# myLabel=Label(root,font="Ariel,100",text="Facial Recognition System",fg="red")
# myLabel.pack()
root.geometry("600x600+0+0")
root.resizable(0,0)
# Add image file
bg = PhotoImage(file = "D:/Project Data_org/image.png")
# Show image using label
label1 = Label( root, image = bg)
label1.place(x = 0, y = 0)

path1 = "D:/Project Data_org"
path = "D:/Project Data_org/test.jpeg" 

def prediction(img):
    import cv2
    from skimage.feature import hog
    import pickle 
    import os
    face_cascade = cv2.CascadeClassifier('C:/Users/halwa/Downloads/haarcascade_frontalface_default.xml')
    model = pickle.load(open("D:/Project Data_org/model_hog.sav", 'rb'))
    

    #image = cv2.imread(img,0)
    img_res = cv2.resize(img, (128,128))
    faces = face_cascade.detectMultiScale(img_res,scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
 # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        roi_color = img_res[y:y + h, x:x + w]
        cv2.imwrite(os.path.join(path1, 'test' + '.jpeg'), roi_color)
        
    

    image_t = cv2.imread(path,0)
    img_res_t = cv2.resize(image_t, (128,128))

   
    fd, hog_image = hog(img_res_t, orientations=9, pixels_per_cell=(8, 8),
                cells_per_block=(2, 2), visualize=True, multichannel=False)

    # Converting into numpy array    
    fd = fd.reshape(1,-1)
    fd.shape
    y_pred = model.predict(fd)
    
    # defining labels
    labels = ["Abdullah","Affan","Ali","Aziz","Basit","Eesha","Eman","Faraz", "Fasih",
                       "Hala","Hamra","Hasan","Hira","Jamali","Jawwad","Laviza","Parshant","Rehmat",
                       "Shehriyar","Subhan", "Wardah"]

    return labels[y_pred[0]]

def real_time():
    import cv2
    face_cascade = cv2.CascadeClassifier('C:/Users/halwa/Downloads/haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0)
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
        for (x, y, w, h) in faces:
            # print(x,y,w,h)
            roi_gray = gray[y:y+h, x:x+w] #(ycord_start, ycord_end)
            roi_color = frame[y:y+h, x:x+w]
            # img_item = "my-image.png"
            # cv2.imwrite(img_item, roi_gray)
            color = (255, 0, 0) #BGR (0-255)
            stroke = 2  #how thick a line will be.
            end_cord_x = x+w
            end_cord_y= y+h
            cv2.rectangle(frame, (x,y), (end_cord_x,end_cord_y),color , stroke)
            pred = prediction(frame)
            cv2.putText(frame,pred, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
            cv2.imshow('frame',frame)
            
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
       
        # When everything done, release the capture
    cap.release()   
    cv2.destroyAllWindows()


def recognize():
    from tkinter.filedialog import askopenfilename
    import cv2
    from skimage.feature import hog
    import pickle 
    import os
    face_cascade = cv2.CascadeClassifier('C:/Users/halwa/Downloads/haarcascade_frontalface_default.xml')
    model = pickle.load(open("D:/Project Data_org/model_hog.sav", 'rb'))
    
    labels = ["Abdullah","Affan","Ali","Aziz","Basit","Eesha","Eman","Faraz", "Fasih",
                       "Hala","Hamra","Hasan","Hira","Jamali","Jawwad","Laviza","Parshant","Rehmat",
                       "Shehriyar","Subhan", "Wardah"]
    path1 = "D:/Project Data_org"
    path = "D:/Project Data_org/test.jpeg" 
    
    filename = askopenfilename()
    test1 = cv2.imread(filename)
    label = prediction(test1)

    output=Label(root,font=",15",text="name: ",fg="red")
    output.pack(pady = 15)
    output.place =  (300,300)
    print(output.config(text=label))

          
button = Button(root, text='Real time Person identification', width=40, command=real_time)
button.pack(pady = 15)
button.pack()
button = Button(root, text='Image identification', width=40, command=recognize)
button.pack(pady = 15)
button.pack()


root.mainloop()

