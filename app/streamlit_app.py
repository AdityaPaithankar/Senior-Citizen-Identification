

with tab_upload:
    uploaded_file = st.file_uploader(
        "Upload an image containing one or more people",
        type=["jpg", "jpeg", "png"],
        key="senior_image_uploader",
    )

with tab_snapshot:
    camera_file = st.camera_input(
        "Capture a photo using your camera",
        key="senior_camera_snapshot",
    )

with tab_live:
    st.subheader("Live Multi-Face Detection")
    st.write(
        "Allow camera access in your browser. The live feed labels each "
        "detected face and writes debounced detection events to the CSV log."
    )
    st.info(
        "Live logging is event-based, not person identification. "
        "It cannot tell whether a person seen later is the same individual."
    )

    webrtc_context = webrtc_streamer(
        key="senior-citizen-live-webcam",
        video_processor_factory=SeniorCitizenVideoProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if webrtc_context.state.playing:
        st.success("Live webcam is running.")
        if webrtc_context.video_processor is not None:
            processor_error = webrtc_context.video_processor.error_message
            if processor_error:
                st.error(f"Video processing error: {processor_error}")


# -------------------- UPLOADED IMAGE / SNAPSHOT PROCESSING --------------------
image_file = uploaded_file if uploaded_file is not None else camera_file

if image_file is not None:
    try:
        image = Image.open(image_file).convert("RGB")
        rgb_image = np.array(image)
        frame_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        with st.spinner("Detecting faces and estimating attributes..."):
            results = pipeline.predict_faces(frame_bgr)

        annotated_bgr = annotate_frame(frame_bgr, results)
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(rgb_image, use_container_width=True)
        with col2:
            st.subheader("Detection Results")
            st.image(annotated_rgb, use_container_width=True)

        if results:
            st.success(f"Detected {len(results)} face(s).")
            result_df = pd.DataFrame(
                [
                    {
                        "Estimated Age": person["age"],
                        "Predicted Gender": person["gender"],
                        "Status": (
                            "Senior Citizen"
                            if person["senior"]
                            else "Non-Senior"
                        ),
                    }
                    for person in results
                ]
            )
            st.dataframe(result_df, use_container_width=True)

            image_bytes = image_file.getvalue()
            # Stable hash within this running Python process is sufficient
            # to prevent duplicate logging caused by ordinary Streamlit reruns.
            image_key = str(hash(image_bytes))
            if st.session_state.get("last_logged_image") != image_key:
                save_logs(results)
                st.session_state["last_logged_image"] = image_key
                st.success("Image detections saved to the CSV log.")
        else:
            st.warning("No faces detected. Try a clear, front-facing image.")

    except Exception as exc:
        st.error(f"Could not process this image: {exc}")


# -------------------- VISITOR LOG --------------------
st.divider()
st.subheader("📋 Detection Log")

if LOG_FILE.exists() and LOG_FILE.stat().st_size > 0:
    try:
        with CSV_LOCK:
            logs_df = pd.read_csv(LOG_FILE)

        st.dataframe(logs_df.tail(100), use_container_width=True)
        st.download_button(
            "Download CSV Log",
            data=LOG_FILE.read_bytes(),
            file_name="visitor_log.csv",
            mime="text/csv",
            key="download_visitor_log",
        )
    except Exception as exc:
        st.warning(f"Could not read the log file: {exc}")
else:
    st.write("No detections have been logged yet.")

st.caption(
    "Age and gender are model estimates, not verified facts. "
    "The senior-status threshold is configured in pipeline.py. "
    "Use an appropriate privacy notice and obtain consent before using "
    "a camera to monitor people."
)
