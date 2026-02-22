namespace SkinSegmentation
{
    partial class MainForm
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            System.Windows.Forms.DataVisualization.Charting.ChartArea chartArea1 = new System.Windows.Forms.DataVisualization.Charting.ChartArea();
            System.Windows.Forms.DataVisualization.Charting.Legend legend1 = new System.Windows.Forms.DataVisualization.Charting.Legend();
            System.Windows.Forms.DataVisualization.Charting.Series series1 = new System.Windows.Forms.DataVisualization.Charting.Series();
            System.Windows.Forms.DataVisualization.Charting.ChartArea chartArea2 = new System.Windows.Forms.DataVisualization.Charting.ChartArea();
            System.Windows.Forms.DataVisualization.Charting.Legend legend2 = new System.Windows.Forms.DataVisualization.Charting.Legend();
            System.Windows.Forms.DataVisualization.Charting.Series series2 = new System.Windows.Forms.DataVisualization.Charting.Series();
            this.pictureBox1 = new System.Windows.Forms.PictureBox();
            this.pictureBox2 = new System.Windows.Forms.PictureBox();
            this.btnPredict = new System.Windows.Forms.Button();
            this.label1 = new System.Windows.Forms.Label();
            this.label2 = new System.Windows.Forms.Label();
            this.btnTrain = new System.Windows.Forms.Button();
            this.txtHxW = new System.Windows.Forms.TextBox();
            this.label5 = new System.Windows.Forms.Label();
            this.pnlImg1 = new System.Windows.Forms.Panel();
            this.pnlImg2 = new System.Windows.Forms.Panel();
            this.btnSavePicBox2 = new System.Windows.Forms.Button();
            this.label7 = new System.Windows.Forms.Label();
            this.label8 = new System.Windows.Forms.Label();
            this.txtNumOfTrainImgs = new System.Windows.Forms.TextBox();
            this.txtTrainTime = new System.Windows.Forms.TextBox();
            this.label9 = new System.Windows.Forms.Label();
            this.txtImgPath = new System.Windows.Forms.TextBox();
            this.btnEvaluate = new System.Windows.Forms.Button();
            this.folderBrowserDialog1 = new System.Windows.Forms.FolderBrowserDialog();
            this.lstAllImgs = new System.Windows.Forms.ListBox();
            this.grpInfo1 = new System.Windows.Forms.GroupBox();
            this.txtTrainResChk = new System.Windows.Forms.TextBox();
            this.grpInfo2 = new System.Windows.Forms.GroupBox();
            this.nudPredictThresh = new System.Windows.Forms.NumericUpDown();
            this.label15 = new System.Windows.Forms.Label();
            this.grpInfo3 = new System.Windows.Forms.GroupBox();
            this.txtEvalResultsChk = new System.Windows.Forms.TextBox();
            this.txtEvalThresh = new System.Windows.Forms.TextBox();
            this.txtEvalTime = new System.Windows.Forms.TextBox();
            this.label11 = new System.Windows.Forms.Label();
            this.label4 = new System.Windows.Forms.Label();
            this.grpAllImgs = new System.Windows.Forms.GroupBox();
            this.grpImgAccuracy = new System.Windows.Forms.GroupBox();
            this.txtImgName = new System.Windows.Forms.TextBox();
            this.label17 = new System.Windows.Forms.Label();
            this.txtPredictionTime = new System.Windows.Forms.TextBox();
            this.label16 = new System.Windows.Forms.Label();
            this.txtImgPrecision = new System.Windows.Forms.TextBox();
            this.txtImgRecall = new System.Windows.Forms.TextBox();
            this.label10 = new System.Windows.Forms.Label();
            this.label14 = new System.Windows.Forms.Label();
            this.grpOverallAccuracy = new System.Windows.Forms.GroupBox();
            this.txtOverallPrecision = new System.Windows.Forms.TextBox();
            this.txtOverallRecall = new System.Windows.Forms.TextBox();
            this.label12 = new System.Windows.Forms.Label();
            this.label13 = new System.Windows.Forms.Label();
            this.txtNumOfEvalImgs = new System.Windows.Forms.TextBox();
            this.label3 = new System.Windows.Forms.Label();
            this.pnlHist1 = new System.Windows.Forms.Panel();
            this.chrtSkinPosterior = new System.Windows.Forms.DataVisualization.Charting.Chart();
            this.chrtNonSkinPosterior = new System.Windows.Forms.DataVisualization.Charting.Chart();
            this.panel1 = new System.Windows.Forms.Panel();
            this.btnSavePicBox3 = new System.Windows.Forms.Button();
            this.pictureBox3 = new System.Windows.Forms.PictureBox();
            this.label6 = new System.Windows.Forms.Label();
            this.saveFileDialog1 = new System.Windows.Forms.SaveFileDialog();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox1)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox2)).BeginInit();
            this.pnlImg1.SuspendLayout();
            this.pnlImg2.SuspendLayout();
            this.grpInfo1.SuspendLayout();
            this.grpInfo2.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.nudPredictThresh)).BeginInit();
            this.grpInfo3.SuspendLayout();
            this.grpAllImgs.SuspendLayout();
            this.grpImgAccuracy.SuspendLayout();
            this.grpOverallAccuracy.SuspendLayout();
            this.pnlHist1.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.chrtSkinPosterior)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.chrtNonSkinPosterior)).BeginInit();
            this.panel1.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox3)).BeginInit();
            this.SuspendLayout();
            // 
            // pictureBox1
            // 
            this.pictureBox1.Location = new System.Drawing.Point(4, 4);
            this.pictureBox1.Margin = new System.Windows.Forms.Padding(4);
            this.pictureBox1.Name = "pictureBox1";
            this.pictureBox1.Size = new System.Drawing.Size(495, 329);
            this.pictureBox1.SizeMode = System.Windows.Forms.PictureBoxSizeMode.StretchImage;
            this.pictureBox1.TabIndex = 0;
            this.pictureBox1.TabStop = false;
            // 
            // pictureBox2
            // 
            this.pictureBox2.Location = new System.Drawing.Point(4, 4);
            this.pictureBox2.Margin = new System.Windows.Forms.Padding(4);
            this.pictureBox2.Name = "pictureBox2";
            this.pictureBox2.Size = new System.Drawing.Size(500, 329);
            this.pictureBox2.SizeMode = System.Windows.Forms.PictureBoxSizeMode.StretchImage;
            this.pictureBox2.TabIndex = 1;
            this.pictureBox2.TabStop = false;
            // 
            // btnPredict
            // 
            this.btnPredict.Enabled = false;
            this.btnPredict.Font = new System.Drawing.Font("Tahoma", 12F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.btnPredict.Location = new System.Drawing.Point(569, 172);
            this.btnPredict.Margin = new System.Windows.Forms.Padding(4);
            this.btnPredict.Name = "btnPredict";
            this.btnPredict.Size = new System.Drawing.Size(169, 100);
            this.btnPredict.TabIndex = 2;
            this.btnPredict.Text = "PREDICT";
            this.btnPredict.UseVisualStyleBackColor = true;
            this.btnPredict.EnabledChanged += new System.EventHandler(this.btnPredict_EnabledChanged);
            this.btnPredict.Click += new System.EventHandler(this.btnPredict_Click);
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Font = new System.Drawing.Font("Tahoma", 12F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label1.Location = new System.Drawing.Point(182, 15);
            this.label1.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(162, 24);
            this.label1.TabIndex = 3;
            this.label1.Text = "Original Image";
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Font = new System.Drawing.Font("Tahoma", 12F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label2.Location = new System.Drawing.Point(1262, 15);
            this.label2.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(197, 24);
            this.label2.TabIndex = 4;
            this.label2.Text = "Segmented Image";
            // 
            // btnTrain
            // 
            this.btnTrain.Font = new System.Drawing.Font("Tahoma", 12F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.btnTrain.Location = new System.Drawing.Point(569, 50);
            this.btnTrain.Margin = new System.Windows.Forms.Padding(4);
            this.btnTrain.Name = "btnTrain";
            this.btnTrain.Size = new System.Drawing.Size(169, 98);
            this.btnTrain.TabIndex = 5;
            this.btnTrain.Text = "TRAIN";
            this.btnTrain.UseVisualStyleBackColor = true;
            this.btnTrain.Click += new System.EventHandler(this.btnTrain_Click);
            // 
            // txtHxW
            // 
            this.txtHxW.Font = new System.Drawing.Font("Tahoma", 9.75F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.txtHxW.Location = new System.Drawing.Point(234, 30);
            this.txtHxW.Margin = new System.Windows.Forms.Padding(4);
            this.txtHxW.Name = "txtHxW";
            this.txtHxW.ReadOnly = true;
            this.txtHxW.Size = new System.Drawing.Size(90, 27);
            this.txtHxW.TabIndex = 11;
            this.txtHxW.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.Font = new System.Drawing.Font("Tahoma", 9.75F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label5.Location = new System.Drawing.Point(186, 33);
            this.label5.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(50, 21);
            this.label5.TabIndex = 12;
            this.label5.Text = "HxW";
            // 
            // pnlImg1
            // 
            this.pnlImg1.AutoScroll = true;
            this.pnlImg1.AutoScrollMinSize = new System.Drawing.Size(1, 1);
            this.pnlImg1.BorderStyle = System.Windows.Forms.BorderStyle.Fixed3D;
            this.pnlImg1.Controls.Add(this.pictureBox1);
            this.pnlImg1.Location = new System.Drawing.Point(16, 50);
            this.pnlImg1.Margin = new System.Windows.Forms.Padding(4);
            this.pnlImg1.Name = "pnlImg1";
            this.pnlImg1.Size = new System.Drawing.Size(512, 348);
            this.pnlImg1.TabIndex = 15;
            // 
            // pnlImg2
            // 
            this.pnlImg2.AutoScroll = true;
            this.pnlImg2.BorderStyle = System.Windows.Forms.BorderStyle.Fixed3D;
            this.pnlImg2.Controls.Add(this.btnSavePicBox2);
            this.pnlImg2.Controls.Add(this.pictureBox2);
            this.pnlImg2.Location = new System.Drawing.Point(1100, 50);
            this.pnlImg2.Margin = new System.Windows.Forms.Padding(4);
            this.pnlImg2.Name = "pnlImg2";
            this.pnlImg2.Size = new System.Drawing.Size(512, 348);
            this.pnlImg2.TabIndex = 16;
            // 
            // btnSavePicBox2
            // 
            this.btnSavePicBox2.Font = new System.Drawing.Font("Tahoma", 7.8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.btnSavePicBox2.Location = new System.Drawing.Point(-2, -2);
            this.btnSavePicBox2.Margin = new System.Windows.Forms.Padding(4);
            this.btnSavePicBox2.Name = "btnSavePicBox2";
            this.btnSavePicBox2.Size = new System.Drawing.Size(55, 26);
            this.btnSavePicBox2.TabIndex = 40;
            this.btnSavePicBox2.Text = "Save";
            this.btnSavePicBox2.UseVisualStyleBackColor = true;
            this.btnSavePicBox2.Click += new System.EventHandler(this.btnSavePicBox2_Click);
            // 
            // label7
            // 
            this.label7.AutoSize = true;
            this.label7.Font = new System.Drawing.Font("Tahoma", 9.75F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label7.Location = new System.Drawing.Point(8, 65);
            this.label7.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label7.Name = "label7";
            this.label7.Size = new System.Drawing.Size(99, 21);
            this.label7.TabIndex = 24;
            this.label7.Text = "Train Time";
            // 
            // label8
            // 
            this.label8.AutoSize = true;
            this.label8.Font = new System.Drawing.Font("Tahoma", 9.75F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label8.Location = new System.Drawing.Point(9, 30);
            this.label8.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label8.Name = "label8";
            this.label8.Size = new System.Drawing.Size(93, 21);
            this.label8.TabIndex = 23;
            this.label8.Text = "# Images";
            // 
            // txtNumOfTrainImgs
            // 
            this.txtNumOfTrainImgs.Font = new System.Drawing.Font("Tahoma", 9.75F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.txtNumOfTrainImgs.Location = new System.Drawing.Point(105, 27);
            this.txtNumOfTrainImgs.Margin = new System.Windows.Forms.Padding(4);
            this.txtNumOfTrainImgs.Name = "txtNumOfTrainImgs";
            this.txtNumOfTrainImgs.ReadOnly = true;
            this.txtNumOfTrainImgs.Size = new System.Drawing.Size(81, 27);
            this.txtNumOfTrainImgs.TabIndex = 22;
            this.txtNumOfTrainImgs.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            // 
            // txtTrainTime
            // 
            this.txtTrainTime.Font = new System.Drawing.Font("Tahoma", 9.75F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.txtTrainTime.Location = new System.Drawing.Point(105, 62);
            this.txtTrainTime.Margin = new System.Windows.Forms.Padding(4);
            this.txtTrainTime.Name = "txtTrainTime";
            this.txtTrainTime.ReadOnly = true;
            this.txtTrainTime.Size = new System.Drawing.Size(219, 27);
            this.txtTrainTime.TabIndex = 21;
            this.txtTrainTime.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            // 
            // label9
            // 
            this.label9.AutoSize = true;
            this.label9.Font = new System.Drawing.Font("Tahoma", 9.75F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label9.Location = new System.Drawing.Point(8, 68);
            this.label9.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label9.Name = "label9";
            this.label9.Size = new System.Drawing.Size(59, 21);
            this.label9.TabIndex = 26;
            this.label9.Text = "Name";
            // 
            // txtImgPath
            // 
            this.txtImgPath.Font = new System.Drawing.Font("Tahoma", 9.75F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.txtImgPath.Location = new System.Drawing.Point(105, 65);
            this.txtImgPath.Margin = new System.Windows.Forms.Padding(4);
            this.txtImgPath.Name = "txtImgPath";
            this.txtImgPath.ReadOnly = true;
            this.txtImgPath.Size = new System.Drawing.Size(219, 27);
            this.txtImgPath.TabIndex = 25;
            this.txtImgPath.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            // 
            // btnEvaluate
            // 
            this.btnEvaluate.Enabled = false;
            this.btnEvaluate.Font = new System.Drawing.Font("Tahoma", 12F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.btnEvaluate.Location = new System.Drawing.Point(569, 298);
            this.btnEvaluate.Margin = new System.Windows.Forms.Padding(4);
            this.btnEvaluate.Name = "btnEvaluate";
            this.btnEvaluate.Size = new System.Drawing.Size(169, 100);
            this.btnEvaluate.TabIndex = 27;
            this.btnEvaluate.Text = "EVALUATE";
            this.btnEvaluate.UseVisualStyleBackColor = true;
            this.btnEvaluate.EnabledChanged += new System.EventHandler(this.btnEvaluate_EnabledChanged);
            this.btnEvaluate.Click += new System.EventHandler(this.btnEvaluate_Click);
            // 
            // lstAllImgs
            // 
            this.lstAllImgs.Font = new System.Drawing.Font("Tahoma", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lstAllImgs.FormattingEnabled = true;
            this.lstAllImgs.HorizontalScrollbar = true;
            this.lstAllImgs.ItemHeight = 17;
            this.lstAllImgs.Location = new System.Drawing.Point(8, 67);
            this.lstAllImgs.Margin = new System.Windows.Forms.Padding(4);
            this.lstAllImgs.Name = "lstAllImgs";
            this.lstAllImgs.ScrollAlwaysVisible = true;
            this.lstAllImgs.Size = new System.Drawing.Size(249, 327);
            this.lstAllImgs.TabIndex = 31;
            this.lstAllImgs.SelectedIndexChanged += new System.EventHandler(this.lstAllImgs_SelectedIndexChanged);
            // 
            // grpInfo1
            // 
            this.grpInfo1.Controls.Add(this.txtTrainResChk);
            this.grpInfo1.Controls.Add(this.txtNumOfTrainImgs);
            this.grpInfo1.Controls.Add(this.txtTrainTime);
            this.grpInfo1.Controls.Add(this.label8);
            this.grpInfo1.Controls.Add(this.label7);
            this.grpInfo1.Enabled = false;
            this.grpInfo1.Location = new System.Drawing.Point(748, 50);
            this.grpInfo1.Margin = new System.Windows.Forms.Padding(4);
            this.grpInfo1.Name = "grpInfo1";
            this.grpInfo1.Padding = new System.Windows.Forms.Padding(4);
            this.grpInfo1.Size = new System.Drawing.Size(332, 98);
            this.grpInfo1.TabIndex = 33;
            this.grpInfo1.TabStop = false;
            this.grpInfo1.Text = "Info";
            this.grpInfo1.VisibleChanged += new System.EventHandler(this.grpInfo1_VisibleChanged);
            // 
            // txtTrainResChk
            // 
            this.txtTrainResChk.Font = new System.Drawing.Font("Tahoma", 9.75F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.txtTrainResChk.ForeColor = System.Drawing.Color.Blue;
            this.txtTrainResChk.Location = new System.Drawing.Point(190, 27);
            this.txtTrainResChk.Margin = new System.Windows.Forms.Padding(4);
            this.txtTrainResChk.Name = "txtTrainResChk";
            this.txtTrainResChk.Size = new System.Drawing.Size(134, 27);
            this.txtTrainResChk.TabIndex = 25;
            this.txtTrainResChk.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            // 
            // grpInfo2
            // 
            this.grpInfo2.Controls.Add(this.nudPredictThresh);
            this.grpInfo2.Controls.Add(this.label15);
            this.grpInfo2.Controls.Add(this.txtImgPath);
            this.grpInfo2.Controls.Add(this.txtHxW);
            this.grpInfo2.Controls.Add(this.label5);
            this.grpInfo2.Controls.Add(this.label9);
            this.grpInfo2.Enabled = false;
            this.grpInfo2.Location = new System.Drawing.Point(748, 172);
            this.grpInfo2.Margin = new System.Windows.Forms.Padding(4);
            this.grpInfo2.Name = "grpInfo2";
            this.grpInfo2.Padding = new System.Windows.Forms.Padding(4);
            this.grpInfo2.Size = new System.Drawing.Size(332, 100);
            this.grpInfo2.TabIndex = 2;
            this.grpInfo2.TabStop = false;
            this.grpInfo2.Text = "Info";
            this.grpInfo2.VisibleChanged += new System.EventHandler(this.grpInfo2_VisibleChanged);
            // 
            // nudPredictThresh
            // 
            this.nudPredictThresh.DecimalPlaces = 3;
            this.nudPredictThresh.Font = new System.Drawing.Font("Tahoma", 10.2F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.nudPredictThresh.Increment = new decimal(new int[] {
            1,
            0,
            0,
            196608});
            this.nudPredictThresh.Location = new System.Drawing.Point(105, 30);
            this.nudPredictThresh.Maximum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.nudPredictThresh.Name = "nudPredictThresh";
            this.nudPredictThresh.Size = new System.Drawing.Size(81, 28);
            this.nudPredictThresh.TabIndex = 29;
            this.nudPredictThresh.ValueChanged += new System.EventHandler(this.nudPredictThresh_ValueChanged);
            // 
            // label15
            // 
            this.label15.AutoSize = true;
            this.label15.Font = new System.Drawing.Font("Tahoma", 9.75F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label15.Location = new System.Drawing.Point(8, 33);
            this.label15.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label15.Name = "label15";
            this.label15.Size = new System.Drawing.Size(95, 21);
            this.label15.TabIndex = 28;
            this.label15.Text = "Threshold";
            // 
            // grpInfo3
            // 
            this.grpInfo3.Controls.Add(this.txtEvalResultsChk);
            this.grpInfo3.Controls.Add(this.txtEvalThresh);
            this.grpInfo3.Controls.Add(this.txtEvalTime);
            this.grpInfo3.Controls.Add(this.label11);
            this.grpInfo3.Controls.Add(this.label4);
            this.grpInfo3.Enabled = false;
            this.grpInfo3.Location = new System.Drawing.Point(748, 298);
            this.grpInfo3.Margin = new System.Windows.Forms.Padding(4);
            this.grpInfo3.Name = "grpInfo3";
            this.grpInfo3.Padding = new System.Windows.Forms.Padding(4);
            this.grpInfo3.Size = new System.Drawing.Size(332, 100);
            this.grpInfo3.TabIndex = 34;
            this.grpInfo3.TabStop = false;
            this.grpInfo3.Text = "Info";
            this.grpInfo3.VisibleChanged += new System.EventHandler(this.grpInfo3_VisibleChanged);
            // 
            // txtEvalResultsChk
            // 
            this.txtEvalResultsChk.Font = new System.Drawing.Font("Tahoma", 9.75F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.txtEvalResultsChk.ForeColor = System.Drawing.Color.Blue;
            this.txtEvalResultsChk.Location = new System.Drawing.Point(190, 25);
            this.txtEvalResultsChk.Margin = new System.Windows.Forms.Padding(4);
            this.txtEvalResultsChk.Name = "txtEvalResultsChk";
            this.txtEvalResultsChk.Size = new System.Drawing.Size(134, 27);
            this.txtEvalResultsChk.TabIndex = 26;
            this.txtEvalResultsChk.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            // 
            // txtEvalThresh
            // 
            this.txtEvalThresh.Font = new System.Drawing.Font("Tahoma", 9.75F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.txtEvalThresh.Location = new System.Drawing.Point(105, 25);
            this.txtEvalThresh.Margin = new System.Windows.Forms.Padding(4);
            this.txtEvalThresh.Name = "txtEvalThresh";
            this.txtEvalThresh.Size = new System.Drawing.Size(82, 27);
            this.txtEvalThresh.TabIndex = 25;
            this.txtEvalThresh.Text = "0";
            this.txtEvalThresh.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            // 
            // txtEvalTime
            // 
            this.txtEvalTime.Font = new System.Drawing.Font("Tahoma", 9.75F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.txtEvalTime.Location = new System.Drawing.Point(105, 60);
            this.txtEvalTime.Margin = new System.Windows.Forms.Padding(4);
            this.txtEvalTime.Name = "txtEvalTime";
            this.txtEvalTime.ReadOnly = true;
            this.txtEvalTime.Size = new System.Drawing.Size(220, 27);
            this.txtEvalTime.TabIndex = 25;
            this.txtEvalTime.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            // 
            // label11
            // 
            this.label11.AutoSize = true;
            this.label11.Font = new System.Drawing.Font("Tahoma", 9.75F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label11.Location = new System.Drawing.Point(9, 28);
            this.label11.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label11.Name = "label11";
            this.label11.Size = new System.Drawing.Size(95, 21);
            this.label11.TabIndex = 26;
            this.label11.Text = "Threshold";
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Font = new System.Drawing.Font("Tahoma", 9.75F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label4.Location = new System.Drawing.Point(9, 62);
            this.label4.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(91, 21);
            this.label4.TabIndex = 26;
            this.label4.Text = "Eval Time";
            // 
            // grpAllImgs
            // 
            this.grpAllImgs.Controls.Add(this.grpImgAccuracy);
            this.grpAllImgs.Controls.Add(this.grpOverallAccuracy);
            this.grpAllImgs.Controls.Add(this.txtNumOfEvalImgs);
            this.grpAllImgs.Controls.Add(this.label3);
            this.grpAllImgs.Controls.Add(this.lstAllImgs);
            this.grpAllImgs.Font = new System.Drawing.Font("Tahoma", 12F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.grpAllImgs.Location = new System.Drawing.Point(539, 406);
            this.grpAllImgs.Margin = new System.Windows.Forms.Padding(4);
            this.grpAllImgs.Name = "grpAllImgs";
            this.grpAllImgs.Padding = new System.Windows.Forms.Padding(4);
            this.grpAllImgs.Size = new System.Drawing.Size(553, 406);
            this.grpAllImgs.TabIndex = 35;
            this.grpAllImgs.TabStop = false;
            this.grpAllImgs.Text = "ALL Images";
            this.grpAllImgs.Visible = false;
            this.grpAllImgs.VisibleChanged += new System.EventHandler(this.grpAllImgs_VisibleChanged);
            // 
            // grpImgAccuracy
            // 
            this.grpImgAccuracy.Controls.Add(this.txtImgName);
            this.grpImgAccuracy.Controls.Add(this.label17);
            this.grpImgAccuracy.Controls.Add(this.txtPredictionTime);
            this.grpImgAccuracy.Controls.Add(this.label16);
            this.grpImgAccuracy.Controls.Add(this.txtImgPrecision);
            this.grpImgAccuracy.Controls.Add(this.txtImgRecall);
            this.grpImgAccuracy.Controls.Add(this.label10);
            this.grpImgAccuracy.Controls.Add(this.label14);
            this.grpImgAccuracy.Location = new System.Drawing.Point(265, 196);
            this.grpImgAccuracy.Margin = new System.Windows.Forms.Padding(4);
            this.grpImgAccuracy.Name = "grpImgAccuracy";
            this.grpImgAccuracy.Padding = new System.Windows.Forms.Padding(4);
            this.grpImgAccuracy.Size = new System.Drawing.Size(276, 198);
            this.grpImgAccuracy.TabIndex = 33;
            this.grpImgAccuracy.TabStop = false;
            this.grpImgAccuracy.Text = "Image Accuracy";
            this.grpImgAccuracy.Visible = false;
            this.grpImgAccuracy.VisibleChanged += new System.EventHandler(this.grpImgAccuracy_VisibleChanged);
            // 
            // txtImgName
            // 
            this.txtImgName.Font = new System.Drawing.Font("Tahoma", 9.75F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.txtImgName.Location = new System.Drawing.Point(109, 42);
            this.txtImgName.Margin = new System.Windows.Forms.Padding(4);
            this.txtImgName.Name = "txtImgName";
            this.txtImgName.ReadOnly = true;
            this.txtImgName.Size = new System.Drawing.Size(163, 27);
            this.txtImgName.TabIndex = 29;
            this.txtImgName.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            // 
            // label17
            // 
            this.label17.AutoSize = true;
            this.label17.Font = new System.Drawing.Font("Tahoma", 9.75F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label17.Location = new System.Drawing.Point(11, 44);
            this.label17.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label17.Name = "label17";
            this.label17.Size = new System.Drawing.Size(99, 21);
            this.label17.TabIndex = 30;
            this.label17.Text = "Img Name";
            // 
            // txtPredictionTime
            // 
            this.txtPredictionTime.Font = new System.Drawing.Font("Tahoma", 9.75F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.txtPredictionTime.Location = new System.Drawing.Point(109, 77);
            this.txtPredictionTime.Margin = new System.Windows.Forms.Padding(4);
            this.txtPredictionTime.Name = "txtPredictionTime";
            this.txtPredictionTime.ReadOnly = true;
            this.txtPredictionTime.Size = new System.Drawing.Size(163, 27);
            this.txtPredictionTime.TabIndex = 27;
            this.txtPredictionTime.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            // 
            // label16
            // 
            this.label16.AutoSize = true;
            this.label16.Font = new System.Drawing.Font("Tahoma", 9.75F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label16.Location = new System.Drawing.Point(11, 79);
            this.label16.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label16.Name = "label16";
            this.label16.Size = new System.Drawing.Size(100, 21);
            this.label16.TabIndex = 28;
            this.label16.Text = "Pred. Time";
            // 
            // txtImgPrecision
            // 
            this.txtImgPrecision.Font = new System.Drawing.Font("Tahoma", 9.75F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.txtImgPrecision.Location = new System.Drawing.Point(109, 128);
            this.txtImgPrecision.Margin = new System.Windows.Forms.Padding(4);
            this.txtImgPrecision.Name = "txtImgPrecision";
            this.txtImgPrecision.ReadOnly = true;
            this.txtImgPrecision.Size = new System.Drawing.Size(163, 27);
            this.txtImgPrecision.TabIndex = 25;
            this.txtImgPrecision.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            // 
            // txtImgRecall
            // 
            this.txtImgRecall.Font = new System.Drawing.Font("Tahoma", 9.75F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.txtImgRecall.Location = new System.Drawing.Point(109, 163);
            this.txtImgRecall.Margin = new System.Windows.Forms.Padding(4);
            this.txtImgRecall.Name = "txtImgRecall";
            this.txtImgRecall.ReadOnly = true;
            this.txtImgRecall.Size = new System.Drawing.Size(163, 27);
            this.txtImgRecall.TabIndex = 8;
            this.txtImgRecall.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            // 
            // label10
            // 
            this.label10.AutoSize = true;
            this.label10.Font = new System.Drawing.Font("Tahoma", 9.75F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label10.Location = new System.Drawing.Point(14, 166);
            this.label10.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label10.Name = "label10";
            this.label10.Size = new System.Drawing.Size(61, 21);
            this.label10.TabIndex = 13;
            this.label10.Text = "Recall";
            // 
            // label14
            // 
            this.label14.AutoSize = true;
            this.label14.Font = new System.Drawing.Font("Tahoma", 9.75F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label14.Location = new System.Drawing.Point(14, 131);
            this.label14.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label14.Name = "label14";
            this.label14.Size = new System.Drawing.Size(88, 21);
            this.label14.TabIndex = 26;
            this.label14.Text = "Precision";
            // 
            // grpOverallAccuracy
            // 
            this.grpOverallAccuracy.Controls.Add(this.txtOverallPrecision);
            this.grpOverallAccuracy.Controls.Add(this.txtOverallRecall);
            this.grpOverallAccuracy.Controls.Add(this.label12);
            this.grpOverallAccuracy.Controls.Add(this.label13);
            this.grpOverallAccuracy.Location = new System.Drawing.Point(269, 67);
            this.grpOverallAccuracy.Margin = new System.Windows.Forms.Padding(4);
            this.grpOverallAccuracy.Name = "grpOverallAccuracy";
            this.grpOverallAccuracy.Padding = new System.Windows.Forms.Padding(4);
            this.grpOverallAccuracy.Size = new System.Drawing.Size(276, 113);
            this.grpOverallAccuracy.TabIndex = 32;
            this.grpOverallAccuracy.TabStop = false;
            this.grpOverallAccuracy.Text = "Overall Accuracy";
            this.grpOverallAccuracy.Visible = false;
            this.grpOverallAccuracy.VisibleChanged += new System.EventHandler(this.grpOverallAccuracy_VisibleChanged);
            // 
            // txtOverallPrecision
            // 
            this.txtOverallPrecision.Font = new System.Drawing.Font("Tahoma", 9.75F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.txtOverallPrecision.Location = new System.Drawing.Point(105, 36);
            this.txtOverallPrecision.Margin = new System.Windows.Forms.Padding(4);
            this.txtOverallPrecision.Name = "txtOverallPrecision";
            this.txtOverallPrecision.ReadOnly = true;
            this.txtOverallPrecision.Size = new System.Drawing.Size(163, 27);
            this.txtOverallPrecision.TabIndex = 25;
            this.txtOverallPrecision.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            // 
            // txtOverallRecall
            // 
            this.txtOverallRecall.Font = new System.Drawing.Font("Tahoma", 9.75F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.txtOverallRecall.Location = new System.Drawing.Point(105, 71);
            this.txtOverallRecall.Margin = new System.Windows.Forms.Padding(4);
            this.txtOverallRecall.Name = "txtOverallRecall";
            this.txtOverallRecall.ReadOnly = true;
            this.txtOverallRecall.Size = new System.Drawing.Size(163, 27);
            this.txtOverallRecall.TabIndex = 8;
            this.txtOverallRecall.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            // 
            // label12
            // 
            this.label12.AutoSize = true;
            this.label12.Font = new System.Drawing.Font("Tahoma", 9.75F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label12.Location = new System.Drawing.Point(10, 74);
            this.label12.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label12.Name = "label12";
            this.label12.Size = new System.Drawing.Size(61, 21);
            this.label12.TabIndex = 13;
            this.label12.Text = "Recall";
            // 
            // label13
            // 
            this.label13.AutoSize = true;
            this.label13.Font = new System.Drawing.Font("Tahoma", 9.75F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label13.Location = new System.Drawing.Point(10, 39);
            this.label13.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label13.Name = "label13";
            this.label13.Size = new System.Drawing.Size(88, 21);
            this.label13.TabIndex = 26;
            this.label13.Text = "Precision";
            // 
            // txtNumOfEvalImgs
            // 
            this.txtNumOfEvalImgs.Font = new System.Drawing.Font("Tahoma", 9.75F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.txtNumOfEvalImgs.Location = new System.Drawing.Point(113, 33);
            this.txtNumOfEvalImgs.Margin = new System.Windows.Forms.Padding(4);
            this.txtNumOfEvalImgs.Name = "txtNumOfEvalImgs";
            this.txtNumOfEvalImgs.ReadOnly = true;
            this.txtNumOfEvalImgs.Size = new System.Drawing.Size(144, 27);
            this.txtNumOfEvalImgs.TabIndex = 27;
            this.txtNumOfEvalImgs.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Font = new System.Drawing.Font("Tahoma", 9.75F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label3.Location = new System.Drawing.Point(8, 36);
            this.label3.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(93, 21);
            this.label3.TabIndex = 28;
            this.label3.Text = "# Images";
            // 
            // pnlHist1
            // 
            this.pnlHist1.Controls.Add(this.chrtSkinPosterior);
            this.pnlHist1.Controls.Add(this.chrtNonSkinPosterior);
            this.pnlHist1.Location = new System.Drawing.Point(16, 406);
            this.pnlHist1.Margin = new System.Windows.Forms.Padding(4);
            this.pnlHist1.Name = "pnlHist1";
            this.pnlHist1.Size = new System.Drawing.Size(512, 406);
            this.pnlHist1.TabIndex = 36;
            this.pnlHist1.Visible = false;
            // 
            // chrtSkinPosterior
            // 
            chartArea1.Name = "ChartArea1";
            this.chrtSkinPosterior.ChartAreas.Add(chartArea1);
            legend1.Name = "Legend1";
            this.chrtSkinPosterior.Legends.Add(legend1);
            this.chrtSkinPosterior.Location = new System.Drawing.Point(8, 4);
            this.chrtSkinPosterior.Margin = new System.Windows.Forms.Padding(4);
            this.chrtSkinPosterior.Name = "chrtSkinPosterior";
            this.chrtSkinPosterior.Palette = System.Windows.Forms.DataVisualization.Charting.ChartColorPalette.None;
            series1.ChartArea = "ChartArea1";
            series1.Legend = "Legend1";
            series1.Name = "Series1";
            this.chrtSkinPosterior.Series.Add(series1);
            this.chrtSkinPosterior.Size = new System.Drawing.Size(495, 198);
            this.chrtSkinPosterior.TabIndex = 17;
            this.chrtSkinPosterior.Text = "Skin Posterior";
            // 
            // chrtNonSkinPosterior
            // 
            chartArea2.Name = "ChartArea1";
            this.chrtNonSkinPosterior.ChartAreas.Add(chartArea2);
            legend2.Name = "Legend1";
            this.chrtNonSkinPosterior.Legends.Add(legend2);
            this.chrtNonSkinPosterior.Location = new System.Drawing.Point(8, 196);
            this.chrtNonSkinPosterior.Margin = new System.Windows.Forms.Padding(4);
            this.chrtNonSkinPosterior.Name = "chrtNonSkinPosterior";
            this.chrtNonSkinPosterior.Palette = System.Windows.Forms.DataVisualization.Charting.ChartColorPalette.None;
            series2.ChartArea = "ChartArea1";
            series2.Legend = "Legend1";
            series2.Name = "Series1";
            this.chrtNonSkinPosterior.Series.Add(series2);
            this.chrtNonSkinPosterior.Size = new System.Drawing.Size(495, 198);
            this.chrtNonSkinPosterior.TabIndex = 18;
            this.chrtNonSkinPosterior.Text = "Green";
            // 
            // panel1
            // 
            this.panel1.AutoScroll = true;
            this.panel1.BorderStyle = System.Windows.Forms.BorderStyle.Fixed3D;
            this.panel1.Controls.Add(this.btnSavePicBox3);
            this.panel1.Controls.Add(this.pictureBox3);
            this.panel1.Location = new System.Drawing.Point(1100, 465);
            this.panel1.Margin = new System.Windows.Forms.Padding(4);
            this.panel1.Name = "panel1";
            this.panel1.Size = new System.Drawing.Size(512, 348);
            this.panel1.TabIndex = 38;
            // 
            // btnSavePicBox3
            // 
            this.btnSavePicBox3.Font = new System.Drawing.Font("Tahoma", 7.8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.btnSavePicBox3.Location = new System.Drawing.Point(-2, -2);
            this.btnSavePicBox3.Margin = new System.Windows.Forms.Padding(4);
            this.btnSavePicBox3.Name = "btnSavePicBox3";
            this.btnSavePicBox3.Size = new System.Drawing.Size(55, 28);
            this.btnSavePicBox3.TabIndex = 41;
            this.btnSavePicBox3.Text = "Save";
            this.btnSavePicBox3.UseVisualStyleBackColor = true;
            this.btnSavePicBox3.Click += new System.EventHandler(this.btnSavePicBox3_Click);
            // 
            // pictureBox3
            // 
            this.pictureBox3.Location = new System.Drawing.Point(4, 4);
            this.pictureBox3.Margin = new System.Windows.Forms.Padding(4);
            this.pictureBox3.Name = "pictureBox3";
            this.pictureBox3.Size = new System.Drawing.Size(500, 329);
            this.pictureBox3.SizeMode = System.Windows.Forms.PictureBoxSizeMode.StretchImage;
            this.pictureBox3.TabIndex = 1;
            this.pictureBox3.TabStop = false;
            // 
            // label6
            // 
            this.label6.AutoSize = true;
            this.label6.Font = new System.Drawing.Font("Tahoma", 12F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label6.Location = new System.Drawing.Point(1251, 430);
            this.label6.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label6.Name = "label6";
            this.label6.Size = new System.Drawing.Size(218, 24);
            this.label6.TabIndex = 37;
            this.label6.Text = "Ground Truth Image";
            // 
            // MainForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(120F, 120F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Dpi;
            this.AutoSize = true;
            this.ClientSize = new System.Drawing.Size(1628, 830);
            this.Controls.Add(this.panel1);
            this.Controls.Add(this.label6);
            this.Controls.Add(this.pnlHist1);
            this.Controls.Add(this.grpAllImgs);
            this.Controls.Add(this.grpInfo3);
            this.Controls.Add(this.grpInfo2);
            this.Controls.Add(this.grpInfo1);
            this.Controls.Add(this.btnEvaluate);
            this.Controls.Add(this.pnlImg2);
            this.Controls.Add(this.pnlImg1);
            this.Controls.Add(this.btnTrain);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.btnPredict);
            this.Margin = new System.Windows.Forms.Padding(4);
            this.MaximizeBox = false;
            this.Name = "MainForm";
            this.Text = "Skin Segmentation";
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox1)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox2)).EndInit();
            this.pnlImg1.ResumeLayout(false);
            this.pnlImg2.ResumeLayout(false);
            this.grpInfo1.ResumeLayout(false);
            this.grpInfo1.PerformLayout();
            this.grpInfo2.ResumeLayout(false);
            this.grpInfo2.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.nudPredictThresh)).EndInit();
            this.grpInfo3.ResumeLayout(false);
            this.grpInfo3.PerformLayout();
            this.grpAllImgs.ResumeLayout(false);
            this.grpAllImgs.PerformLayout();
            this.grpImgAccuracy.ResumeLayout(false);
            this.grpImgAccuracy.PerformLayout();
            this.grpOverallAccuracy.ResumeLayout(false);
            this.grpOverallAccuracy.PerformLayout();
            this.pnlHist1.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.chrtSkinPosterior)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.chrtNonSkinPosterior)).EndInit();
            this.panel1.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox3)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.PictureBox pictureBox1;
        private System.Windows.Forms.PictureBox pictureBox2;
        private System.Windows.Forms.Button btnPredict;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.Button btnTrain;
        private System.Windows.Forms.TextBox txtHxW;
        private System.Windows.Forms.Label label5;
        private System.Windows.Forms.Panel pnlImg1;
        private System.Windows.Forms.Panel pnlImg2;
        private System.Windows.Forms.Label label7;
        private System.Windows.Forms.Label label8;
        private System.Windows.Forms.TextBox txtNumOfTrainImgs;
        private System.Windows.Forms.TextBox txtTrainTime;
        private System.Windows.Forms.Label label9;
        private System.Windows.Forms.TextBox txtImgPath;
        private System.Windows.Forms.Button btnEvaluate;
        private System.Windows.Forms.FolderBrowserDialog folderBrowserDialog1;
        private System.Windows.Forms.ListBox lstAllImgs;
        private System.Windows.Forms.GroupBox grpInfo1;
        private System.Windows.Forms.GroupBox grpInfo2;
        private System.Windows.Forms.GroupBox grpInfo3;
        private System.Windows.Forms.GroupBox grpAllImgs;
        private System.Windows.Forms.TextBox txtEvalTime;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.GroupBox grpImgAccuracy;
        private System.Windows.Forms.TextBox txtImgPrecision;
        private System.Windows.Forms.TextBox txtImgRecall;
        private System.Windows.Forms.Label label10;
        private System.Windows.Forms.Label label14;
        private System.Windows.Forms.GroupBox grpOverallAccuracy;
        private System.Windows.Forms.TextBox txtOverallPrecision;
        private System.Windows.Forms.TextBox txtOverallRecall;
        private System.Windows.Forms.Label label12;
        private System.Windows.Forms.Label label13;
        private System.Windows.Forms.TextBox txtNumOfEvalImgs;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.Label label15;
        private System.Windows.Forms.TextBox txtEvalThresh;
        private System.Windows.Forms.Label label11;
        private System.Windows.Forms.NumericUpDown nudPredictThresh;
        private System.Windows.Forms.Panel pnlHist1;
        private System.Windows.Forms.DataVisualization.Charting.Chart chrtSkinPosterior;
        private System.Windows.Forms.DataVisualization.Charting.Chart chrtNonSkinPosterior;
        private System.Windows.Forms.TextBox txtTrainResChk;
        private System.Windows.Forms.TextBox txtEvalResultsChk;
        private System.Windows.Forms.Panel panel1;
        private System.Windows.Forms.PictureBox pictureBox3;
        private System.Windows.Forms.Label label6;
        private System.Windows.Forms.Button btnSavePicBox2;
        private System.Windows.Forms.Button btnSavePicBox3;
        private System.Windows.Forms.SaveFileDialog saveFileDialog1;
        private System.Windows.Forms.TextBox txtImgName;
        private System.Windows.Forms.Label label17;
        private System.Windows.Forms.TextBox txtPredictionTime;
        private System.Windows.Forms.Label label16;
    }
}

