import t2v_metrics
clip_flant5_score = t2v_metrics.VQAScore(model='clip-flant5-xxl') # our recommended scoring model

### For a single (image, text) pair
image = "output_cvpr/coco17_pretrained_sd35/generated_images_coco17_g1/000000000139.png" # an image path in string format
text = "someone talks on the phone angrily while another person sits happily"
score = clip_flant5_score(images=[image], texts=[text])