import gibbs_sampler
import numpy as np
import image_utils


def question_7():
    clean_image = image_utils.get_scaled_image(
        image_utils.get_image_from_location("../data/denoising_images/im_clean.png"))
    noisy_image = image_utils.get_scaled_image(
        image_utils.get_image_from_location("../data/denoising_images/im_noisy.png"))

    weight_param = 0.3
    bias_param = np.array(noisy_image) * 0.5
    t_max = 100

    images = gibbs_sampler.gibbs_sampler(t_max, noisy_image.copy(), (bias_param.copy(), weight_param))
    image = np.average(images, axis=0)

    image_utils.save_grid_image_to_location(image_utils.get_scaled_noisy_image(image),
                                            "../data/denoised_images/im_fexed_denoised.png", "L")

    print 'Mean per-pixel error', np.average(np.absolute(clean_image.flatten() - image.flatten()))


def question_8():
    clean_image = image_utils.get_scaled_image(
        image_utils.get_image_from_location("../data/denoising_images/im_clean.png"))
    noisy_image = image_utils.get_scaled_image(
        image_utils.get_image_from_location("../data/denoising_images/im_noisy.png"))

    params = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    bias_term = 0.7
    weight_param = 0.9
    # for bias_term in params:
    # for weight_param in params:

    print 'Bias: ', bias_term, ", Weight: ", weight_param
    bias_param = np.array(noisy_image) * bias_term
    t_max = 100

    images = gibbs_sampler.gibbs_sampler(t_max, noisy_image.copy(), (bias_param.copy(), weight_param))
    image = np.average(images, axis=0)

    image_utils.save_grid_image_to_location(image_utils.get_scaled_noisy_image(image),
                                            "../data/denoised_images/im_variable_denoised_scaled" + str(
                                                bias_term) + "_" + str(weight_param) + ".png", "L")

    print 'Mean per-pixel error', np.average(np.absolute(clean_image.flatten() - image.flatten()))


def main():
    # question_7()
    question_8()


if __name__ == '__main__':
    main()
