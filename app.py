import streamlit as st
import matplotlib.pyplot as plt
import preprocess
import model

st.title("DeepLabV3 Liver Tumor Segmentation")
image_nii = st.file_uploader('Выберите снимок КТ', type=['nii'])
mask_nii = st.file_uploader('Выберите маску', type=['nii'])

if 'images' not in st.session_state or 'preds' not in st.session_state:
    st.session_state['images'], st.session_state['preds'] = [], []

def preprocess_file(file, file_name, is_mask=False):
    with open(f'temp/{file_name}.nii', 'wb') as f:
        f.write(file.read())

    slices = preprocess.get_nii_slices(f'temp/{file_name}.nii')
    if not is_mask:
        slices = preprocess.normalize_nii(slices)

    return slices

if st.button('Обработать снимок'):
    if image_nii is not None:
        st.session_state['images'] = preprocess_file(image_nii, 'image')
        st.session_state['preds'] = model.prediction(st.session_state['images'])
    else:
        st.error('Убедитесь, что загрузили снимок')

if len(st.session_state['images']) > 0:
    slice_index = st.slider('Срезы', 0, len(st.session_state['images']) - 1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    ax[0].imshow(st.session_state['images'][slice_index], cmap='gray')
    ax[0].set_title('Исходное изображение')
    ax[0].axis('off')

    ax[1].imshow(st.session_state['preds'][slice_index].squeeze(0), cmap='gray')
    ax[1].set_title('Маска')
    ax[1].axis('off')

    st.pyplot(fig)

if st.button('Рассчитать Индекс Жаккара'):
    if image_nii is not None and mask_nii is not None:
        if len(st.session_state['preds']) > 0:
            iou = model.iou(st.session_state['preds'], preprocess_file(mask_nii, 'mask', is_mask=True))
            st.write('Индекс Жаккара:', iou)
        else:
            st.error('Сначала нужно обработать снимок')
    else:
        st.error('Убедитесь, что загрузили и снимок, и маску')
    
    