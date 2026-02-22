using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Reflection.Emit;
using System.Text;
using System.Windows.Forms;
using System.Windows.Forms.DataVisualization.Charting;
using static System.Net.Mime.MediaTypeNames;
using static SkinSegmentation.SkinSegmentation;

namespace SkinSegmentation
{
    public partial class MainForm : Form
    {
        public MainForm()
        {
            InitializeComponent();
        }

        #region Global Data
        string curPathOfClrImgs, curPathOfMaskImgs;
        BayesModel bayesModel;
        const int numOfFeatures = 360;
        #endregion

        #region EVENT HANDLERS

        #region Buttons Click
        private void btnTrain_Click(object sender, EventArgs e)
        {
            try
            {
                string folderPath = string.Empty;
                string[] imgPaths = null, maskPaths = null;
                folderBrowserDialog1.Description = "Please Select the Folder of COLORED IMAGES:";
                if (folderBrowserDialog1.ShowDialog() == DialogResult.OK)
                {
                    folderPath = folderBrowserDialog1.SelectedPath;
                    List<string> files = GetAllImageFiles(folderPath);
                    files.Sort();
                    imgPaths = files.ToArray();
                    if (imgPaths.Length == 0)
                    {
                        MessageBox.Show("No valid images are found inside the specified folder");
                        return;
                    }
                }
                else
                {
                    return;
                }
                folderBrowserDialog1.Description = "Please Select the Folder of MASK IMAGES:";
                if (folderBrowserDialog1.ShowDialog() == DialogResult.OK)
                {
                    string maskFolderPath = folderBrowserDialog1.SelectedPath;
                    List<string> files = GetAllImageFiles(maskFolderPath);
                    files.Sort();
                    maskPaths = files.ToArray();
                }
                else
                {
                    return;
                }
                if (!ValidateImageMaskPairs(imgPaths, maskPaths))
                {
                    MessageBox.Show("Mismatch between images & corresponding masks");
                    return;
                }

                //Update UI
                {
                    grpInfo1.Enabled = false;
                    grpInfo2.Enabled = false;
                    grpInfo3.Enabled = false;
                    btnPredict.Enabled = false;
                    btnEvaluate.Enabled = false;
                    grpAllImgs.Visible = false;
                    pnlHist1.Visible = false;

                    lstAllImgs.Items.Clear();
                }

                Stopwatch sw = Stopwatch.StartNew();
                bayesModel = Train(imgPaths, maskPaths);
                sw.Stop();
                txtTrainTime.Text = sw.Elapsed.ToString();
                txtNumOfTrainImgs.Text = imgPaths.Length.ToString();

                string filePath = folderPath + "\\TrainedModel.dat";
                if (File.Exists(filePath))
                {
                    BayesModel savedModel = LoadTrainedModel(filePath);
                    int numOfWrong = CompareModels(bayesModel, savedModel);
                    if (numOfWrong > 0)
                    {
                        MessageBox.Show("WRONG OUTPUT in " + numOfWrong + " Cases! check console for details");
                        txtTrainResChk.ForeColor = Color.Red;
                        txtTrainResChk.Text = "WRONG";
                    }
                    else
                    {
                        txtTrainResChk.ForeColor = Color.Green;
                        txtTrainResChk.Text = "CORRECT";
                    }
                }
                else
                {
                    txtTrainResChk.ForeColor = Color.Blue;
                    txtTrainResChk.Text = "CALCULATED & SAVED";
                    SaveTrainedModel(bayesModel, folderPath, "TrainedModel.dat");
                }

                double[] skinPosterior = new double[numOfFeatures];
                double[] nonskinPosterior = new double[numOfFeatures];
                for (int i = 0; i < numOfFeatures; i++)
                {
                    skinPosterior[i] = bayesModel.PriorSkin * bayesModel.LikelihoodSkin[i];
                    nonskinPosterior[i] = bayesModel.PriorNonSkin * bayesModel.LikelihoodNonSkin[i];
                }
                ImageOperations.DrawHistogram(skinPosterior, chrtSkinPosterior, Color.Red, "Skin Posterior");
                ImageOperations.DrawHistogram(nonskinPosterior, chrtNonSkinPosterior, Color.Blue, "Non-Skin Posterior");

                //Update UI
                {
                    btnPredict.Enabled = true;
                    btnEvaluate.Enabled = true;
                    pnlHist1.Visible = true;
                    grpInfo1.Enabled = true;
                    grpInfo2.Enabled = true;
                    grpInfo3.Enabled = true;
                }

            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
        }
        private void btnPredict_Click(object sender, EventArgs e)
        {
            OpenFileDialog openFileDialog1 = new OpenFileDialog();
            if (openFileDialog1.ShowDialog() == DialogResult.OK)
            {
                //update UI
                pictureBox1.Image = null;
                pictureBox2.Image = null;
                pictureBox3.Image = null;

                //Open the browsed image and display it
                string OpenedFilePath = openFileDialog1.FileName;
                Bitmap bm = new Bitmap(OpenedFilePath);
                pictureBox1.Image = bm;
                txtHxW.Text = bm.Height.ToString() + "x" + bm.Width.ToString();
                txtImgPath.Text = OpenedFilePath;
                double threshold = (double)nudPredictThresh.Value;
                try
                {
                    Stopwatch sw = Stopwatch.StartNew();
                    RGBPixel[,] segmentedImage = Predict(OpenedFilePath, bayesModel, threshold);
                    sw.Stop();

                    ImageOperations.DisplayImage(segmentedImage, pictureBox2);
                }
                catch (Exception ex)
                {
                    MessageBox.Show(ex.Message);
                }
            }
        }
        private void btnEvaluate_Click(object sender, EventArgs e)
        {
            try
            {
                string[] imgPaths = null, maskPaths = null;
                folderBrowserDialog1.Description = "Please Select the Folder of COLORED IMAGES:";
                if (folderBrowserDialog1.ShowDialog() == DialogResult.OK)
                {
                    curPathOfClrImgs = folderBrowserDialog1.SelectedPath;
                    List<string> files = GetAllImageFiles(curPathOfClrImgs);
                    files.Sort();
                    imgPaths = files.ToArray();
                    if (imgPaths.Length == 0)
                    {
                        MessageBox.Show("No valid images are found inside the specified folder");
                        return;
                    }
                }
                else
                {
                    return;
                }
                folderBrowserDialog1.Description = "Please Select the Folder of MASK IMAGES:";
                if (folderBrowserDialog1.ShowDialog() == DialogResult.OK)
                {
                    curPathOfMaskImgs = folderBrowserDialog1.SelectedPath;
                    List<string> files = GetAllImageFiles(curPathOfMaskImgs);
                    files.Sort();
                    maskPaths = files.ToArray();
                }
                else
                {
                    return;
                }
                if (!ValidateImageMaskPairs(imgPaths, maskPaths))
                {
                    MessageBox.Show("Mismatch between images & corresponding masks");
                    curPathOfMaskImgs = curPathOfClrImgs = string.Empty;
                    return;
                }

                lstAllImgs.Items.Clear();
                grpOverallAccuracy.Visible = false;
                grpImgAccuracy.Visible = false;

                double thresh = double.Parse(txtEvalThresh.Text);
                Stopwatch sw = Stopwatch.StartNew();
                Metrics result = Evaluate(imgPaths, maskPaths, bayesModel, thresh);
                sw.Stop();

                txtEvalTime.Text = sw.Elapsed.ToString();
                txtNumOfEvalImgs.Text = imgPaths.Length.ToString();


                string filePath = curPathOfMaskImgs + "\\" + thresh.ToString();
                filePath += "_Metrics.dat";
                if (File.Exists(filePath))
                {
                    Metrics expectedMetrics = LoadEvaluationMetrics(filePath);
                    bool correct = CompareEvalMetrics(result, expectedMetrics);
                    if (!correct)
                    {
                        MessageBox.Show("WRONG OUTPUT! check console for details");
                        txtEvalResultsChk.ForeColor = Color.Red;
                        txtEvalResultsChk.Text = "WRONG";
                    }
                    else
                    {
                        txtEvalResultsChk.ForeColor = Color.Green;
                        txtEvalResultsChk.Text = "CORRECT";
                    }
                }
                else
                {
                    txtEvalResultsChk.ForeColor = Color.Blue;
                    txtEvalResultsChk.Text = "CALCULATED & SAVED";
                    SaveEvaluationMetrics(result, filePath);
                }

                txtOverallPrecision.Text = Math.Round(result.Precision,2).ToString();
                txtOverallRecall.Text = Math.Round(result.Recall, 2).ToString();

                for (int i = 0; i < imgPaths.Length; i++)
                {
                    lstAllImgs.Items.Add(imgPaths[i] + "$" + maskPaths[i]);
                }
                grpAllImgs.Visible = true;
                grpOverallAccuracy.Visible = true;
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            } 
        }
        private void btnSavePicBox2_Click(object sender, EventArgs e)
        {
            if (saveFileDialog1.ShowDialog() == DialogResult.OK)
            {
                string filePath = saveFileDialog1.FileName;
                pictureBox2.Image.Save(filePath, ImageFormat.Bmp);
            }
        }
        private void btnSavePicBox3_Click(object sender, EventArgs e)
        {
            if (saveFileDialog1.ShowDialog() == DialogResult.OK)
            {
                string filePath = saveFileDialog1.FileName;
                pictureBox3.Image.Save(filePath, ImageFormat.Bmp);
            }
        }
        #endregion

        #region Listboxes Select Changed
        private void lstAllImgs_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (lstAllImgs.SelectedIndex == -1)
            {
                return;
            }
            grpImgAccuracy.Visible = true;

            string[] paths = lstAllImgs.Items[lstAllImgs.SelectedIndex].ToString().Split('$');
            string imgPath = paths[0];
            string maskPath = paths[1]; 
            // Load the image
            Bitmap bm = new Bitmap(imgPath);
            pictureBox1.Image = bm;

            try
            {
                double evalThresh = double.Parse(txtEvalThresh.Text);
                
                Stopwatch sw = Stopwatch.StartNew();
                RGBPixel[,] segmentedImage = Predict(imgPath, bayesModel, evalThresh);
                Metrics imgMetrics = Evaluate(new string[] { imgPath }, new string[] { maskPath }, bayesModel, evalThresh);
                sw.Stop();

                RGBPixel[,] maskImage = ImageOperations.ApplyMask(imgPath, maskPath);
                
                ImageOperations.DisplayImage(segmentedImage, pictureBox2);
                ImageOperations.DisplayImage(maskImage, pictureBox3);

                txtImgName.Text = ExtractFileName(imgPath);
                txtPredictionTime.Text = sw.Elapsed.ToString();
                txtImgPrecision.Text = Math.Round(imgMetrics.Precision,2).ToString();
                txtImgRecall.Text = Math.Round(imgMetrics.Recall,2).ToString();
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }
        }
        #endregion

        #region Other Handlers for UI
        private void btnPredict_EnabledChanged(object sender, EventArgs e)
        {
            if (!btnPredict.Enabled)
            {
                txtHxW.Text = string.Empty;
                txtImgPath.Text = string.Empty;
            }
            grpInfo2.Visible = btnPredict.Enabled;
        }
        private void btnEvaluate_EnabledChanged(object sender, EventArgs e)
        {
            if (!btnPredict.Enabled)
            {
                txtEvalTime.Text = string.Empty;
            }
            grpInfo3.Visible = btnEvaluate.Enabled;
        }
        private void grpAllImgs_VisibleChanged(object sender, EventArgs e)
        {
            pictureBox1.Image = null;
            pictureBox2.Image = null;
            pictureBox3.Image = null;
        }
        private void nudPredictThresh_ValueChanged(object sender, EventArgs e)
        {
            if (txtImgPath.Text == string.Empty)
                return;

            double threshold = (double)nudPredictThresh.Value;
            RGBPixel[,] segmentedImage = Predict(txtImgPath.Text, bayesModel, threshold);
            RGBPixel[,] coloredImage = ImageOperations.OpenImage(txtImgPath.Text);

            ImageOperations.DisplayImage(coloredImage, pictureBox1);
            ImageOperations.DisplayImage(segmentedImage, pictureBox2);
            pictureBox3.Image = null;
        }
        private void grpInfo3_VisibleChanged(object sender, EventArgs e)
        {
            if (!grpInfo3.Visible)
            {
                txtEvalResultsChk.Text = string.Empty;
                txtEvalTime.Text = string.Empty;
            }
        }
        private void grpInfo2_VisibleChanged(object sender, EventArgs e)
        {
            if (!grpInfo2.Visible)
            {
                txtHxW.Text = string.Empty;
                txtImgPath.Text = string.Empty;
            }
        }
        private void grpInfo1_VisibleChanged(object sender, EventArgs e)
        {
            if (!grpInfo1.Visible)
            {
                txtTrainResChk.Text = string.Empty;
                txtTrainTime.Text = string.Empty;
                txtNumOfTrainImgs.Text = string.Empty;
            }
        }
        private void grpImgAccuracy_VisibleChanged(object sender, EventArgs e)
        {
            if (!grpImgAccuracy.Visible)
            {
                txtImgName.Text = string.Empty;
                txtPredictionTime.Text = string.Empty;
                txtImgPrecision.Text = string.Empty;
                txtImgRecall.Text = string.Empty;
            }
        }
        private void grpOverallAccuracy_VisibleChanged(object sender, EventArgs e)
        {
            if (!grpOverallAccuracy.Visible)
            {
                txtOverallPrecision.Text = string.Empty;
                txtOverallRecall.Text = string.Empty;
            }
        }
        #endregion

        #endregion

        #region Helper Functions
        private bool ValidateImageMaskPairs(string[] imgPaths, string[] maskPaths)
        {
            int N = imgPaths.Length;
            if (N != maskPaths.Length)
            {
                MessageBox.Show("number of images != number of masks");
                return false;
            }
            for (int i = 0; i < N; i++)
            {
                if (ExtractFileName(imgPaths[i]) != ExtractFileName(maskPaths[i]))
                {
                    MessageBox.Show($"Mismatch! image: {imgPaths[i]}, mask: {maskPaths[i]}");
                    return false;
                }
            }
            return true;
        }
        private string ExtractFileName(string path)
        {
            string[] tokens = path.Split('\\');
            string fileName = tokens[tokens.Length - 1].Split('.')[0];
            return fileName;
        }
        private List<string> GetAllImageFiles(string folderPath)
        {
            // Define common image file extensions
            string[] imageExtensions = new string[] { "*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif", "*.tiff", "*.ico" };

            List<string> imageFiles = new List<string>();

            foreach (string extension in imageExtensions)
            {
                // Get all files with the current extension in the folder
                string[] files = Directory.GetFiles(folderPath, extension);

                // Add the found files to the imageFiles list
                imageFiles.AddRange(files);
            }

            return imageFiles;
        }
        private void SaveTrainedModel(BayesModel model, string folderPath, string fileName)
        {
            Stream s = new FileStream(folderPath + "\\" + fileName, FileMode.Create);
            var sw = new BinaryWriter(s);

            //Priors
            sw.Write(model.PriorSkin);
            sw.Write(model.PriorNonSkin);
            
            //Likelihoods
            for (int i = 0; i < numOfFeatures; i++)
            {
                sw.Write(model.LikelihoodSkin[i]);
            }
            for (int i = 0; i < numOfFeatures; i++)
            {
                sw.Write(model.LikelihoodNonSkin[i]);
            }


            s.Close();
            sw.Close();
        }
        private BayesModel LoadTrainedModel(string filePath)
        {
            Stream s = new FileStream(filePath, FileMode.Open);
            var sr = new BinaryReader(s);
            BayesModel model = new BayesModel();
            model.LikelihoodSkin = new double[numOfFeatures];
            model.LikelihoodNonSkin = new double[numOfFeatures];


            //Priors
            model.PriorSkin = sr.ReadDouble();
            model.PriorNonSkin = sr.ReadDouble();
            
            //Likelihoods
            for (int i = 0; i < numOfFeatures; i++)
            {
                model.LikelihoodSkin[i] = sr.ReadDouble();
            }
            for (int i = 0; i < numOfFeatures; i++)
            {
                model.LikelihoodNonSkin[i] = sr.ReadDouble();
            }


            s.Close();
            sr.Close();

            return model;
        }
        private int CompareModels(BayesModel calculatedModel, BayesModel loadedModel)
        {
            string msg = string.Empty;
            int numOfWrong = 0;
            bool correct = true;
            int fpp = 7;
            msg += $"Priors:\n";

            msg += $"   1. Prior Skin: ";
            if (Math.Round(calculatedModel.PriorSkin, fpp) != Math.Round(loadedModel.PriorSkin, fpp))
            {
                msg += $"Mismatch! Calculated {calculatedModel.PriorSkin} Expected {loadedModel.PriorSkin}\n";
                numOfWrong++;
            }
            else
            {
                msg += $"Correct\n";
            }

            msg += $"   2. Prior Non-Skin: ";
            if (Math.Round(calculatedModel.PriorNonSkin, fpp) != Math.Round(loadedModel.PriorNonSkin, fpp))
            {
                msg += $"Mismatch! Calculated {calculatedModel.PriorNonSkin} Expected {loadedModel.PriorNonSkin}\n";
                numOfWrong++;
            }
            else
            {
                msg += $"Correct\n";
            }

            for (int i = 0; i < numOfFeatures; i++)
            {
                msg += $"Hue[{i}]:\n";

                
                if (Math.Round(calculatedModel.LikelihoodSkin[i], fpp) != Math.Round(loadedModel.LikelihoodSkin[i], fpp))
                {
                    msg += $"Likelihood Skin Mismatch! Calculated {calculatedModel.LikelihoodSkin[i]} Expected {loadedModel.LikelihoodSkin[i]}\n";
                    correct = false;
                }
                if (Math.Round(calculatedModel.LikelihoodNonSkin[i], fpp) != Math.Round(loadedModel.LikelihoodNonSkin[i], fpp))
                {
                    msg += $"Likelihood Non Skin Mismatch! Calculated {calculatedModel.LikelihoodNonSkin[i]} Expected {loadedModel.LikelihoodNonSkin[i]}\n";
                    correct = false;
                }
                if (correct)
                {
                    msg += "CORRECT\n";
                }
                else
                {
                    msg += "WRONG\n";
                    numOfWrong++;
                }
            }

            Console.WriteLine(msg);
            return numOfWrong;
        }
        private void SaveEvaluationMetrics(Metrics evalMetrics, string filePath)
        {
            Stream s = new FileStream(filePath, FileMode.Create);
            var sw = new BinaryWriter(s);
            sw.Write(evalMetrics.TP);
            sw.Write(evalMetrics.TN);
            sw.Write(evalMetrics.FP);
            sw.Write(evalMetrics.FN);

            s.Close();
            sw.Close();
        }
        private Metrics LoadEvaluationMetrics(string filePath)
        {
            Metrics evalMetrics = new Metrics();
            Stream s = new FileStream(filePath, FileMode.Open);
            var sr = new BinaryReader(s);
            evalMetrics.TP = sr.ReadInt32();
            evalMetrics.TN = sr.ReadInt32();
            evalMetrics.FP = sr.ReadInt32();
            evalMetrics.FN = sr.ReadInt32();

            s.Close();
            sr.Close();

            return evalMetrics;
        }
        private bool CompareEvalMetrics(Metrics calculatedMetrics, Metrics expectedMetrics)
        {
            string msg = string.Empty;
            int fpp = 5;
            if (Math.Round(calculatedMetrics.Precision, fpp) != Math.Round(expectedMetrics.Precision, fpp))
            {
                msg += $"Precision Mismatch! Calculated {calculatedMetrics.Precision} Expected {expectedMetrics.Precision}\n";
                return false;
            }
            if (Math.Round(calculatedMetrics.Recall, fpp) != Math.Round(expectedMetrics.Recall, fpp))
            {
                msg += $"Recall Mismatch! Calculated {calculatedMetrics.Recall} Expected {expectedMetrics.Recall}\n";
                return false;
            }
            
            Console.WriteLine(msg);
            return true;
        }
        #endregion

    }
}