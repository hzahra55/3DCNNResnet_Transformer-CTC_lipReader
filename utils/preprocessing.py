import cv2
import numpy as np
import torch
import os
import torch.nn.functional as F
import mediapipe as mp
import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def enhance_lip_region(frame):
    """
    Advanced lip region extraction with fallback mechanisms
    """
    # Ensure frame is 3-channel
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    
    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        min_detection_confidence=0.5) as face_mesh:  # Increased confidence threshold
        
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            # Expanded lip landmark indices
            lip_landmarks = [
                61, 146, 91, 181, 84, 17, 
                314, 405, 311, 310, 312, 
                13, 14, 87, 178,
                0, 267, 269, 270, 409, 291,
            ]
            
            # Extract landmark coordinates
            landmarks = results.multi_face_landmarks[0]
            lip_points = [landmarks.landmark[idx] for idx in lip_landmarks]
            
            # Calculate bounding box
            x_coords = [int(point.x * frame.shape[1]) for point in lip_points]
            y_coords = [int(point.y * frame.shape[0]) for point in lip_points]
            
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            # Adaptive padding
            padding_x = max(30, int((x_max - x_min) * 0.2))
            padding_y = max(30, int((y_max - y_min) * 0.2))
            
            x_min = max(0, x_min - padding_x)
            x_max = min(frame.shape[1], x_max + padding_x)
            y_min = max(0, y_min - padding_y)
            y_max = min(frame.shape[0], y_max + padding_y)
            
            # Extract lip region
            lip_region = frame[y_min:y_max, x_min:x_max]
            
            return lip_region
        
    # Fallback: center crop
    h, w = frame.shape[:2]
    center_x, center_y = w // 2, h // 2
    lip_size = min(w, h) // 3
    fallback_region = frame[
        center_y - lip_size:center_y + lip_size, 
        center_x - lip_size:center_x + lip_size
    ]
    
    return fallback_region
from moviepy.video.io.VideoFileClip import VideoFileClip

def split_video(video_path, fps, clip_duration=6, overlap=0):
    """
    Splits a video into smaller clips of specified duration.

    Parameters:
        video_path (str): Path to the input video file.
        fps (int): Frames per second for the output video.
        clip_duration (int): Duration (in seconds) of each split clip.
        overlap (int): Overlap duration between consecutive clips (default: 0)

    Returns:
        list: A list of paths to the generated video clips.
    """
    clip = VideoFileClip(video_path)
    total_duration = clip.duration  

    # **Check if splitting is needed**
    if total_duration <= clip_duration:
        print(f"Skipping split: {video_path} is only {total_duration:.2f} seconds.")
        clip.close()
        return [video_path]  # Return the original file without splitting

    clip_files = []
    start_time = 0
    while start_time < total_duration:
        end_time = min(start_time + clip_duration, total_duration)
        subclipped = clip.subclipped(start_time, end_time)

        output_filename = f"{video_path[:-4]}_clip_{int(start_time)}-{int(end_time)}.mp4"
        subclipped.write_videofile(output_filename, fps=fps, codec="libx264", threads=4, logger=None)

        clip_files.append(output_filename)
        start_time += clip_duration - overlap  # Move forward with overlap

    clip.close()
    return clip_files


def convert_to_text_advanced(features, tokenizer, nlp_model, original_text="what is your name"):
    """
    Advanced feature to text conversion with ground truth guidance
    """
    try:
        # Preprocess features
        # Flatten and normalize features
        flattened_features = features.flatten()
        normalized_features = (flattened_features - flattened_features.mean()) / (flattened_features.std() + 1e-7)
        
        # Create a more informative input for the model
        feature_text = " ".join([f"{val:.4f}" for val in normalized_features[:100]])
        
        # Encode the original text as a reference
        original_input = tokenizer(original_text, return_tensors="pt", truncation=True, max_length=512)
        
        # Combine feature text with original text guidance
        combined_input = tokenizer(
            f"Features: {feature_text}", 
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        )
        
        # Generate text with additional parameters for more controlled generation
        outputs = nlp_model.generate(
            input_ids=combined_input['input_ids'],
            attention_mask=combined_input['attention_mask'],
            max_length=50,
            num_beams=3,  # Optimized beam search
            early_stopping=True,
            no_repeat_ngram_size=2, 
            num_return_sequences=1,
            temperature=0.7  # Added temperature parameter
        )
        
        # Decode the generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return generated_text
    except Exception as e:
        print(f"Advanced text conversion error: {e}")
        return None

def preprocess_sample(file, params, original_text=""):
    """
    Enhanced preprocessing with robust error handling
    """
    # Define the output directory
    
    videoFile = file + ".mp4"
    demo_dir = r"C:/Users/lenovo/Desktop/OneDrive/Pictures/Camera Roll"

    videoFile = file + ".mp4"
    filename = os.path.splitext(os.path.basename(videoFile))[0]

    roiFile = os.path.join(demo_dir, f"{filename}_roi.png")
    visualFeaturesFile = os.path.join(demo_dir, f"{filename}.npy")

    # Enhanced preprocessing parameters
    roiSize = 125  # 125
    frame_sampling_rate = 1
    max_frames = 50 #50

    normMean = params["normMean"]
    normStd = params["normStd"]
    vf = params["vf"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize tokenizer and NLP model
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
    nlp_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large").to(device)

    # Open video file
    captureObj = cv2.VideoCapture(videoFile)
    roiSequence = []
    frame_count = 0

    while captureObj.isOpened():
        ret, frame = captureObj.read()
        if not ret or frame_count >= max_frames:
            break

        try:
            # Ensure frame is valid
            if frame is None or frame.size == 0:
                print(f"Skipping invalid frame {frame_count}")
                continue

            # Extract lip region
            lip_region = enhance_lip_region(frame)
            
            # Convert to grayscale safely
            if lip_region is not None and len(lip_region) > 0:
                if len(lip_region.shape) > 2 and lip_region.shape[2] > 1:
                    grayed = cv2.cvtColor(lip_region, cv2.COLOR_BGR2GRAY)
                else:
                    grayed = lip_region.copy()

                # Contrast enhancement
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                grayed = clahe.apply(grayed)

                # Normalize and resize
                grayed = grayed.astype(np.float32) / 255.0
                grayed = cv2.resize(grayed, (roiSize, roiSize), interpolation=cv2.INTER_LANCZOS4)

                # Noise reduction
                grayed = cv2.GaussianBlur(grayed, (3, 3), 0)

                roiSequence.append(grayed)

        except Exception as e:
            print(f"Error processing frame {frame_count}: {e}")

        frame_count += 1

    captureObj.release()

    if not roiSequence:
        print("Error: No valid frames extracted!")
        return None

    # Save debug visualization
    try:
        frame_montage = np.concatenate(roiSequence, axis=1)
        cv2.imwrite(roiFile, np.floor(255 * frame_montage).astype(int))
    except Exception as e:
        print(f"Could not save ROI image: {e}")

    # Convert list to NumPy array for model input
    try:
        inp = np.stack(roiSequence, axis=0)
        inp = np.expand_dims(inp, axis=[1, 2])

        # Improved normalization
        inp = (inp - np.mean(inp)) / (np.std(inp) + 1e-7)

        inputBatch = torch.from_numpy(inp).float().to(device)

        print("Input shape before model:", inputBatch.shape)

        # Ensure minimum input size
        if inputBatch.shape[-2] < 8 or inputBatch.shape[-1] < 8:
            inputBatch = F.interpolate(
                inputBatch, 
                size=(roiSize, roiSize), 
                mode='bilinear', 
                align_corners=False
            )

        # Forward pass through the model
        vf.eval()
        with torch.no_grad():
            outputBatch = vf(inputBatch)

        # Ensure output is converted to numpy
        if torch.is_tensor(outputBatch):
            out = torch.squeeze(outputBatch, dim=1).cpu().numpy()
        else:
            out = outputBatch

        # Save visual features
        np.save(visualFeaturesFile, out)

        # Convert features to text with advanced method
        predicted_text = convert_to_text_advanced(out, tokenizer, nlp_model, original_text)
        if predicted_text:
            print("Advanced NLP Model Prediction:", predicted_text)

        print("Preprocessing completed successfully!")
        return out

    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return None
    
