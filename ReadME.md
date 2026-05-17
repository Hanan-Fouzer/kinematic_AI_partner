🏋️ Kinematic AI Partner: Bio-Feedback Dashboard


Welcome to the Kinematic AI Partner. This system utilizes a custom-trained RTMPose-m model to provide automated kinematic feedback for weightlifting. By analyzing movement at 30 FPS, the system offers insights into form deviations, rep consistency, and physical intensity.


📋 PrerequisitesOperating System:

 Linux (Ubuntu 20.04/22.04 recommended) or Windows with WSL2.
 Hardware: Minimum 8GB RAM.
 
 Note on GPU: The system is "Hardware-Agnostic." It will prioritize an NVIDIA GPU if available; otherwise, it will automatically switch to CPU-Stable Mode to ensure a crash-free experience.
 
 Python Manager: Miniconda or Anaconda must be installed.
 
 
 🛠️ Step 1: 
 Environment SetupWe use a Conda environment to ensure all library versions (PyTorch, MMPose, Streamlit) match perfectly.
 
 Open your Terminal/Command Prompt.
 
 Navigate to the project directory:cd path/to/Kinematic_AI_Partner_Final

Create the environment from the provided file: conda env create -f environment.yml

Activate the environment:conda activate workout_pose


📂 Step 2: Project StructureThe project is structured to be "Plug-and-Play."

 Ensure the following folders are present in the root directory:mmpose/: 
 
 The core AI framework.work_dirs/: Contains the trained weights (epoch_50.pth)
 .reference_videos/: Contains the expert .json movement signatures
 .app.py: The main dashboard script
 .core_utils.py: The mathematical engine for normalization and DTW.
 
 
 🚀 Step 3: Launching the Dashboard To start the application, run the following command. 
 
 
 The --server.fileWatcherType none flag is included to optimize performance on Linux systems.
 streamlit run app.py --server.fileWatcherType none


💡 How to Use the SystemUpload Video: 

Drag and drop an MP4 or MOV file of a workout (e.g., Bicep Curl or Bench Press).Analysis: 
The system will process the video at a standardized 30 FPS.
Review Results:
Form Deviations: 
View the top 5 frames where your form differed most from the expert. Green dots indicate correct form (< 0.15 error), while Red dots highlight areas for improvement.Stability Graph: Tracks your movement consistency. A dropping line indicates potential muscle fatigue.Repetition Pulse: A waveform showing the rhythm and count of your reps.Notifications: Look for automated alerts regarding your Workout Tempo and Nutritional Needs (Protein recommendations).


⚠️ TroubleshootingCUDA Errors: 

If a "CUDA unknown error" appears in the terminal, the app will continue to function on the CPU. No action is required.Workout Detection: The system matches workouts based on temporal rhythm. For best results, ensure the camera is stable and the lifter's full range of motion is visible.


👨‍💻 AuthorHanan - Final Year Project