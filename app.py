import streamlit as st
import cv2
import numpy as np
from PIL import Image
from rembg import remove
import io

def upscale_image(img):
    return cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

def enhance_quality(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_eq = cv2.equalizeHist(l)
    lab_eq = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

def color_grading(img):
    lookup = np.interp(np.arange(256), [0, 128, 255], [0, 180, 255]).astype('uint8')
    return cv2.LUT(img, lookup)

def remove_bg(img):
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    no_bg = remove(pil_img)
    return cv2.cvtColor(np.array(no_bg), cv2.COLOR_RGBA2BGRA)

def apply_hdr(img):
    hdr = cv2.detailEnhance(img, sigma_s=12, sigma_r=0.15)
    return hdr

def portrait_mode(img):
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    center_x, center_y = w // 2, h // 2
    radius = min(w, h) // 3
    cv2.circle(mask, (center_x, center_y), radius, 255, -1)
    mask = cv2.GaussianBlur(mask, (51, 51), 0)
    blurred = cv2.GaussianBlur(img, (21, 21), 0)
    result = np.where(mask[:, :, None] > 127, img, blurred)
    return result

def apply_all_styles(img):
    img = upscale_image(img)
    img = enhance_quality(img)
    img = color_grading(img)
    img = apply_hdr(img)
    img = portrait_mode(img)
    return img

def main():
    st.title("EternaFrame - One-Click Image Style Editor")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        st.image(image, caption='Original Image', channels="BGR")

        if st.button("Apply All Styles"):
            styled = apply_all_styles(image)
            st.image(styled, caption="Styled Image", channels="BGR")

        if st.button("Remove Background"):
            bg_removed = remove_bg(image)
            st.image(bg_removed, caption="Background Removed", channels="BGRA")

if __name__ == "__main__":
    main()