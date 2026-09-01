from algo import jade, otsu_variance, kapur_entropy, tsallis_entropy, do_multilevel_thresholding
import cv2


def main():
    image_path = "..\\BDS500\\BDS500\\img1.png"
        
    image = cv2.imread(image_path)
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
    histogram_normalized = cv2.normalize(histogram, None, alpha = 1, beta = 0, norm_type=cv2.NORM_L1)
    best_thresholds=jade(histogram=histogram_normalized, NP=50, num_thresholds=3, n_generations=100, c = 0.1, p = 0.05, objective=tsallis_entropy)

    thresholded = do_multilevel_thresholding(image, best_thresholds)

    cv2.imshow("Thresholded image", thresholded)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()