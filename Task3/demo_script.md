# Live demo script (about 4-5 minutes; record as video or GIF)

Preparation: `python Task3/app.py`, open http://127.0.0.1:7860, leave the mode on **Auto**.
Use the *Examples* table under the buttons to fill the inputs with one click, then press **Run**
(or type the prompt yourself, which looks better on video).

| # | What to do | What to say | Expected outcome |
|---|---|---|---|
| 0 | Show the page and the status line | "One text box, images, and a tool-use trace on the right." | Status: Task 2 model found / not found |
| 1 | Type **"Calibrate the camera and undistort this image"**, upload `Task1/Chestboard_data/left01.jpg`, Run | "The router recognised *calibrate* from the words *calibrate* and *undistort*." | Trace: tool `calibrate`, confidence 1.00; gallery original vs undistorted (curved board edges become straight); K with fx 536, RMS 0.41 px |
| 2 | Type **"Add salt and pepper noise and denoise it with a median filter size 5"**, upload `Task1/noise_data/lena.jpg`, Run | "Parameters are read from the sentence: noise type, filter, size 5." | Tool `filter`; original / noisy / filtered; table: median 30.89 dB, SSIM 0.856 (noisy image 18.21 dB) |
| 3 | Change the text to **"compare box, gaussian, median and bilateral denoising"** (keep gaussian noise via *Advanced settings -> Add noise first*), Run | "Same tool, now all four filters side by side." | 4 filtered images and a PSNR/SSIM table |
| 4 | Type **"Detect edges with canny"**, same image, Run; then **"show the high-pass filter in the frequency domain"** | "Different words, different tool: *edges*." | Canny edge map with percentage of edge pixels; then spectrum + high-pass result |
| 5 | Type **"Match the box with the scene using SIFT, ratio 0.75"**, upload `box.png` as image 1 and `box_in_scene.png` as image 2, Run | "Two images, so the router also sees a hint for matching." | Tool `features`; green quadrilateral around the box; 604 / 969 keypoints, 80 matches, 75 inliers (93.8 %) |
| 6 | Re-run with **"... using ORB"** | "ORB is a binary descriptor: usually faster, fewer matches." | 46 good matches, 43 inliers; compare the time with SIFT (ORB is usually faster) |
| 7 | Type **"How big is this object? It is 50 cm away"**, upload `Task3/samples/measure_demo.png`, Run | "Pinhole model: size = pixels x Z / f, f from our calibration." | Tool `measure`; box drawn with about 281 x 188 mm; assumptions listed |
| 8 | Upload a fruit photo (e.g. from `Task2/data/raw/Original Image/...`), type **"Is this fruit fresh or rotten?"**, Run | "Now it calls the Task 2 CNN." | Tool `classify`; class, confidence, Grad-CAM, top-3 bar. If the model is not trained yet: a friendly message, no crash |
| 9 | Type **"hello"**, Run | "It does not guess when it is unsure." | "I need a bit more information" + the list of things it can do; trace shows *No tool chosen* |
| 10 | Switch the mode to **Noise and filtering**, type nothing, Run | "Manual mode skips the router." | Tool `filter` with defaults, trace says "Chosen manually" |
| 11 | (Optional) Upload 5+ files from `Task1/Chestboard_data` to *Extra images*, type **"Recalibrate the camera from my chessboard images"**, Run | "Recalibration with the Task 1 code." | Detected corners, new K and RMS |

Tips for the recording: zoom the browser to 110 %, keep the trace box visible in every shot, and say aloud
which keywords triggered the decision.
