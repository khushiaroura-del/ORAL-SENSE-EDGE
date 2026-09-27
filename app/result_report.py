# ==========================================
# ORAL-SENSE EDGE
# SCREENING RESULT REPORT
# ==========================================

print("--------------------------------")
print("       ORAL-SENSE EDGE")
print("     SCREENING RESULT")
print("--------------------------------")


# ------------------------------------------
# Demo values
# ------------------------------------------

image_quality = "GOOD IMAGE"

predicted_class = "benign_lesions"

model_probability = 97.48

gradcam_available = True


# ------------------------------------------
# Display result
# ------------------------------------------

print()
print("IMAGE QUALITY")
print("--------------------------------")
print(image_quality)


print()
print("AI SCREENING RESULT")
print("--------------------------------")

print(
    "Predicted class:",
    predicted_class
)

print(
    "Model probability:",
    f"{model_probability:.2f}%"
)


# ------------------------------------------
# Explanation
# ------------------------------------------

print()
print("EXPLAINABLE AI")
print("--------------------------------")

if gradcam_available:

    print(
        "Grad-CAM explanation: AVAILABLE"
    )

    print(
        "The heatmap shows regions that "
        "influenced the model prediction."
    )

else:

    print(
        "Grad-CAM explanation: NOT AVAILABLE"
    )


# ------------------------------------------
# Guidance
# ------------------------------------------

print()
print("SCREENING GUIDANCE")
print("--------------------------------")

print(
    "This result is intended only as "
    "a screening aid."
)

print(
    "It should not be used as a medical "
    "diagnosis."
)

print(
    "If you have a persistent or concerning "
    "oral lesion, seek evaluation from a "
    "qualified healthcare professional."
)


# ------------------------------------------
# Privacy
# ------------------------------------------

print()
print("PRIVACY")
print("--------------------------------")

print(
    "Prototype processing is performed "
    "locally on the device."
)


# ------------------------------------------
# Final
# ------------------------------------------

print()
print("--------------------------------")
print("REPORT COMPLETE")
print("--------------------------------")