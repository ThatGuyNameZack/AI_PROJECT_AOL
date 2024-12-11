            avg_confidences = {
                emotion: (np.mean(confidences) if confidences else 0) 
                for emotion, confidences in emotion_confidences.items()
            }
            
            # Determine final emotion with highest average confidence
            final_emotion = max(avg_confidences, key=avg_confidences.get)
            final_confidence = avg_confidences[final_emotion]

            # Detailed logging instead of printing
            logging.info("\n--- Emotion Prediction ---")
            for label, prob in zip(emotion_labels, emotion_probs):
                logging.info(f"{label}: {prob:.2f}")
            logging.info(f"\nCurrent Emotion: {final_emotion}")
            logging.info(f"Confidence: {final_confidence:.2f}")
            logging.info("Average Confidences: %s", 
                  {emotion: f"{conf:.2f}" for emotion, conf in avg_confidences.items()})

            # Draw rectangle around face and display emotion
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
            
            # Set camera resolution
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Width
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Height

            # Adjust other properties
            cam.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)  # Brightness (0.0 to 1.0)
            cam.set(cv2.CAP_PROP_CONTRAST, 0.5)    # Contrast (0.0 to 1.0)
            
            # Color code the text based on emotion
            color_map = {
                'Engaged': (0, 255, 0),      # Green
                'Confused': (255, 165, 0),   # Orange
                'Frustrated': (0, 0, 255),   # Red
                'Bored': (128, 128, 128),    # Gray
                'Drowsy': (255, 0, 255),     # Magenta
                'Distracted': (255, 255, 0) # Yellow
            }
            
            # Use final emotion for display
            display_text = f"{final_emotion} ({final_confidence:.2f})"
            cv2.putText(frame, display_text, (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
                        color_map.get(final_emotion, (255, 255, 255)), 2)

        # Display the frame
        cv2.imshow('Real-Time Emotion Detection', frame)

        # Break loop on 'p' key press
        if cv2.waitKey(1) & 0xFF == ord('p'):
            break

except KeyboardInterrupt:
    logging.info("\nExiting program...")

finally:
    # Release resources
    cam.release()
    cv2.destroyAllWindows()

# Print final emotion statistics with more robust handling
logging.info("\n--- Emotion Detection Summary ---")
emotion_avg_confidences = []
for emotion, confidences in emotion_confidences.items():
    # Use numpy's nanmean to handle potential empty lists
    avg_conf = np.nanmean(confidences) if confidences else 0
    logging.info(f"{emotion}: Average Confidence = {avg_conf:.2f}")
    emotion_avg_confidences.append(avg_conf)

# Create bar graph with error handling
try:
    plt.figure(figsize=(10, 6))
    plt.bar(emotion_labels, emotion_avg_confidences, 
            color=['green', 'orange', 'red', 'gray', 'magenta', 'yellow'])
    plt.title('Emotion Detection - Average Confidences', fontsize=15)
    plt.xlabel('Emotions', fontsize=12)
    plt.ylabel('Average Confidence', fontsize=12)
    plt.ylim(0, 1)  # Set y-axis from 0 to 1

    # Add value labels on top of each bar
    for i, v in enumerate(emotion_avg_confidences):
        plt.text(i, v + 0.01, f'{v:.2f}', ha='center', fontsize=10)

    plt.tight_layout()
    plt.show()
except Exception as e:
    logging.error(f"Error creating bar graph: {e}")