# Program To Read video
# and Extract Frames
import cv2
  
# Function to extract frames
def FrameCapture(path):
      
    # Path to video file
    vidObj = cv2.VideoCapture(path)
  
    # Used as counter variable
    count = 1
    actual_count = 201
    success, image = vidObj.read()
    # number of frames to skip
    numFrameToSave = 3
    
    length = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT))
    print("The length of the video is %d" % length)
    length = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT) / numFrameToSave)
    print("Estimated number of frames at the output %d" % length)
   
    while success:
        # vidObj object calls read
        # function extract frames
        success, image = vidObj.read()
        if not success:
            break
        # Saves the frames with frame-count
        if (count % numFrameToSave == 0):
            cv2.imwrite("C:\\Users\\Darkhan\\source\\repos\\master-thesis-equipment-detection-docs\\pumps\\dataset_name_rendered\\JPEGImages\\frame%d.jpg" % actual_count, image)
            print("Saved image %d" % actual_count)
            actual_count += 1
        count += 1
        
        if actual_count > 1000:
            print("Enough data..")

            break
            
        if cv2.waitKey(10) == 27:                     
            break
  
# Driver Code
if __name__ == '__main__':
  
    # Calling the function
    FrameCapture("C:\\Users\\Darkhan\\Videos\\Captures\\3.mp4")