using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using static SkinSegmentation.ImageOperations;

namespace SkinSegmentation
{
    public struct BayesModel
    {
        public double[] LikelihoodSkin;
        public double[] LikelihoodNonSkin;
        public double PriorSkin;
        public double PriorNonSkin;
    }
    public struct Metrics
    {
        public int TP, TN, FP, FN;
        public double Precision
        {
            get { return 100.0 * (double)TP / (TP + FP); } 
        }

        public double Recall
        {
            get { return 100.0 * (double)TP / (TP + FN); }
        }
    }
    public class SkinSegmentation
    {
        /// <summary>
        /// Train the Bayes model using the given colored images and their corresponding skin mask images
        /// </summary>
        /// <param name="imagePaths">path of the training colored images</param>
        /// <param name="maskPaths">path of the corresponding mask images</param>
        /// <returns>Bayes model (Likelihood & Prior of each class)</returns>
        public static BayesModel Train(string[] imagePaths, string[] maskPaths)
        {
            BayesModel ans = new BayesModel();
            ans.LikelihoodSkin = new double[361];
            ans.LikelihoodNonSkin = new double[361];

            int[] SkinHue = new int[361];
            int[] nonSkinHue = new int[361];


            int numOfSkinPixels = 0;
            int numOfNonSkinPixels = 0;

            object lck = new object();

            Parallel.For(0, imagePaths.Length, i =>
            {
                RGBPixel[,] img = OpenImage(imagePaths[i]);
                RGBPixel[,] mask = OpenImage(maskPaths[i]);

                int height = img.GetLength(0);
                int width = img.GetLength(1);

                int[] localSkinHue = new int[361];
                int[] localNonSkinHue = new int[361];
                int localnumOfSkinPixels = 0;
                int localnumOfNonSkinPixels = 0;

                for (int j = 0; j < height; ++j)
                {
                    for (int k = 0; k < width; ++k)
                    {
                        HSVPixel imagePX = ConvertRgbToHsv(img[j, k]);

                        if (mask[j, k].red == 0)
                        {
                            localNonSkinHue[imagePX.hue]++;
                            localnumOfNonSkinPixels++;
                        }
                        else
                        {
                            localSkinHue[imagePX.hue]++;
                            localnumOfSkinPixels++;
                        }
                    }
                }
                lock (lck)
                {
                    for (int t = 0; t <= 360; t++)
                    {
                        SkinHue[t] += localSkinHue[t];
                        nonSkinHue[t] += localNonSkinHue[t];
                    }
                    numOfSkinPixels += localnumOfSkinPixels;
                    numOfNonSkinPixels += localnumOfNonSkinPixels;
                }
            });

            for (int i = 0; i <= 360; ++i)
            {
                ans.LikelihoodSkin[i] = (double)SkinHue[i] / (double)numOfSkinPixels;
                ans.LikelihoodNonSkin[i] = (double)nonSkinHue[i] / (double)numOfNonSkinPixels;
            }

            ans.PriorSkin = (numOfSkinPixels / (double)(numOfSkinPixels + numOfNonSkinPixels));
            ans.PriorNonSkin = (numOfNonSkinPixels / (double)(numOfSkinPixels + numOfNonSkinPixels));

            return ans;
        }

        /// <summary>
        /// Predict the skin pixels of the given image using the given Bayes Model
        /// </summary>
        /// <param name="imgPath">path of the test image</param>
        /// <param name="model">trained Bayes model</param>
        /// <param name="threshold">threshold of skin posterior</param>
        /// <returns>segmented image</returns>
        public static RGBPixel[,] Predict(string imgPath, BayesModel model, double threshold = 0)
        {
            RGBPixel[,] img = OpenImage(imgPath);
            int height = img.GetLength(0);
            int width = img.GetLength(1);
            RGBPixel[,] ans = new RGBPixel[height, width];

            Parallel.For(0, height, i =>
            {
                for (int j = 0; j < width; ++j)
                {
                    HSVPixel imagePX = ConvertRgbToHsv(img[i, j]);

                    double skinPosterior = model.LikelihoodSkin[imagePX.hue] * model.PriorSkin;
                    double nonSkinPosterior = model.LikelihoodNonSkin[imagePX.hue] * model.PriorNonSkin;

                    if (skinPosterior > nonSkinPosterior && skinPosterior > threshold)
                    {
                        ans[i, j].red = 255;
                        ans[i, j].green = 255;
                        ans[i, j].blue = 255;
                    }
                    else
                    {
                        ans[i, j].red = 0;
                        ans[i, j].green = 0;
                        ans[i, j].blue = 0;
                    }
                }
            });

            return ans;
        }

        /// <summary>
        /// Evaluate the given Bayes model using the given set of images and their corresponding ground-thruth masks
        /// </summary>
        /// <param name="imagePaths"></param>
        /// <param name="maskPaths"></param>
        /// <param name="threshold"></param>
        /// <returns>Evaluation metrics (Precision & Recall)</returns>
        public static Metrics Evaluate(string[] imagePaths, string[] maskPaths, BayesModel trainedModel, double threshold = 0) 
        {
            Metrics ans = new Metrics();
            ans.TP = 0; ans.TN = 0; ans.FP = 0; ans.FN = 0;

            object lck = new object();

            Parallel.For(0, imagePaths.Length, i =>
            {
                RGBPixel[,] predicted = Predict(imagePaths[i], trainedModel, threshold);
                RGBPixel[,] actual = OpenImage(maskPaths[i]);
                int height = predicted.GetLength(0);
                int width = predicted.GetLength(1);
                int localTP = 0, localTN = 0, localFP = 0, localFN = 0;

                for (int j = 0; j<height; ++j)
                {
                    for (int k = 0; k < width; ++k)
                    {
                        if (predicted[j, k].red == 255 && actual[j, k].red == 255)
                            localTP++;
                        else if (predicted[j, k].red == 0 && actual[j, k].red == 0)
                            localTN++;
                        else if (predicted[j, k].red == 255 && actual[j, k].red == 0)
                            localFP++;
                        else if (predicted[j, k].red == 0 && actual[j, k].red == 255)
                            localFN++;
                    }
                }
                lock (lck)
                {
                    ans.TP += localTP;
                    ans.TN += localTN;
                    ans.FP += localFP;
                    ans.FN += localFN;
                }
            });
            return ans;
        }
    }
}
