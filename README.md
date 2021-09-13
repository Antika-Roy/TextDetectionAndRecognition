# Text Detection And Recognition in Natural Scene Images

This is an upgraded work of my undergrad thesis project on "Text detection from natural scene images".
I have used the images(Image dataset was personally collected from the natural scene from Chittagong,Bangladesh by me) to show both text detection with the EAST method and text recognition with Tesseract 4.My image set contains natural scene text detection challenges(descrirbed below) and EAST text detector can successfully detected the bounding boxes having the text.<br>
 
Text detection:<br>
OpenCV’s EAST(Efficient and Accurate Scene Text Detection ) text detector is a deep learning model, based on a novel architecture and training pattern. It is capable of running at near real-time at 13 FPS on 720p images and obtains state-of-the-art text detection accuracy.Also EAST is quite robust, capable of localizing text even when it’s blurred, reflective, or partially obscured.
Link to paper : https://arxiv.org/abs/1704.03155v2<br>

There are many natural scene text detection challenges that have been described by Celine Mancas-Thillou and Bernard Gosselin in their excellent 2017 paper, Natural Scene Text Understanding below:<br>

Image/sensor noise: Sensor noise from a handheld camera is typically higher than that of a traditional scanner. Additionally, low-priced cameras will typically interpolate the pixels of raw sensors to produce real colors.<br>

Viewing angles: Natural scene text can naturally have viewing angles that are not parallel to the text, making the text harder to recognize. Blurring: Uncontrolled environments tend to have blur, especially if the end user is utilizing a smartphone that does not have some form of stabilization.<br>

Lighting conditions: We cannot make any assumptions regarding our lighting conditions in natural scene images. It may be near dark, the flash on the camera may be on, or the sun may be shining brightly, saturating the entire image.<br>

Resolution: Not all cameras are created equal — we may be dealing with cameras with sub-par resolution.<br>

Non-paper objects: Most, but not all, paper is not reflective (at least in context of paper you are trying to scan). Text in natural scenes may be reflective, including logos, signs, etc.<br>

Non-planar objects: Consider what happens when you wrap text around a bottle — the text on the surface becomes distorted and deformed. While humans may still be able to easily “detect” and read the text, our algorithms will struggle. We need to be able to handle such use cases.<br>

Unknown layout: We cannot use any a priori information to give our algorithms “clues” as to where the text resides.<br>

Text Recognition :<br>
Once I have detected the bounding boxes having the text, the next step is to recognize text.<br>

Tesseract<br>
 As per wikipedia-In 2006, Tesseract was considered one of the most accurate open-source OCR engines then available.
The capability of the Tesseract was mostly limited to structured text data. It would perform quite poorly in unstructured text with significant noise. Further development in tesseract has been sponsored by Google since 2006.
Deep-learning based method performs better for the unstructured data. Tesseract 4 added deep-learning based capability with LSTM network(a kind of Recurrent Neural Network) based OCR engine which is focused on the line recognition but also supports the legacy Tesseract OCR engine of Tesseract 3 which works by recognizing character patterns.

Results :<br>
The code uses OpenCV EAST model for text detection and tesseract for text recognition. PSM for the Tesseract has been set accordingly to the image. It is important to note that Tesseract normally requires a clear image for working well.<br>
In the current implementation, I did not consider rotating bounding boxes due to its complexity to implement. But in the real scenario where the text is rotated, the above code will not work well. Also, whenever the image is not very clear, tesseract will have difficulty to recognize the text properly.
We can not expect the OCR model to be 100 % accurate. Still, I have achieved good results with the EAST model and Tesseract with images having natural scene challenges. Adding more filters for processing the image would help in improving the performance of the model.


This is an extension version of my undergrad thesis project on "Text detection from natural scene images".<br>
How to run:<br>
1.Download/clone project<br>
2.open Anaconda promt<br>
3.run in cmd-> python gui.py<br>
4.run with sample images in the project<br>
