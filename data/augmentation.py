import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform
import cv2
class Skeletonization(ImageOnlyTransform):
    def __init__(
        self,
        always_apply=True,
        p=1.0,
    ):
        super(Skeletonization, self).__init__(always_apply, p)

    def apply(self, image, **params):
        image = 255 - image
        image[image > 0] = 255
        image = cv2.ximgproc.thinning(image)
        image = 255 - image
        return image

if __name__ == '__main__':
    im = cv2.imread('/content/images/all/0x69d8/ETL8G_010799.png', 0)
    print(im.shape)
    ske = A.Compose([Skeletonization()])
    im = ske(image = im)['image']
    im[im < 255] = 0
    cv2.imwrite('debug_ske.png', im)