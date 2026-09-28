def get_referral_guidance(predicted_class, model_probability):
    """
    Provides general screening guidance.
    This is NOT a medical diagnosis.
    """

    predicted_class = predicted_class.lower()

    if "malignant" in predicted_class:
        guidance = "Consider professional dental/oral examination."
        level = "FOLLOW-UP RECOMMENDED"

    elif "benign" in predicted_class:
        guidance = "No urgent referral suggested by this screening result. "
        guidance += "Continue routine oral health monitoring."
        level = "ROUTINE MONITORING"

    else:
        guidance = "Consider professional examination for further evaluation."
        level = "PROFESSIONAL REVIEW"

    return {
        "level": level,
        "guidance": guidance,
        "disclaimer": (
            "This tool provides screening assistance only. "
            "It does not provide a medical diagnosis."
        )
    }


if __name__ == "__main__":

    print("=" * 55)
    print("ORAL-SENSE EDGE")
    print("REFERRAL GUIDANCE")
    print("=" * 55)

    result = get_referral_guidance(
        "malignant_lesions",
        85.0
    )

    print("Guidance Level:", result["level"])
    print("Guidance:", result["guidance"])
    print("Disclaimer:", result["disclaimer"])