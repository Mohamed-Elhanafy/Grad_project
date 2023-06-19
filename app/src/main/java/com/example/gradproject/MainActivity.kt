package com.example.gradproject

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Color
import android.opengl.Visibility
import android.os.Bundle
import android.provider.MediaStore
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.example.gradproject.ml.Model
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.ResizeOp.ResizeMethod
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer


lateinit var bitmap: Bitmap
const val REQUEST_CODE = 1
var maxIdx =0

class MainActivity : AppCompatActivity() {

    lateinit var predictBtn: Button
    lateinit var gallery:android.widget.Button
    lateinit var imageView: ImageView
    lateinit var result: TextView


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        predictBtn = findViewById(R.id.button)
        gallery = findViewById(R.id.button2)
        result = findViewById(R.id.result)
        imageView = findViewById(R.id.imageView)

        var imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(176,176 , ResizeMethod.BILINEAR))
            .build()

        gallery.setOnClickListener {
            val cameraIntent =
                Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
            startActivityForResult(cameraIntent, REQUEST_CODE)
        }
        
        predictBtn.setOnClickListener {
            var tensorImage  = TensorImage(DataType.FLOAT32)
            tensorImage.load(bitmap)
            imageProcessor.process(tensorImage)

            val model = Model.newInstance(this)

            // Creates inputs for reference.
            val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 176, 176, 3), DataType.FLOAT32)
            inputFeature0.loadBuffer(tensorImage.buffer)

            // Runs model inference and gets result.
            val outputs = model.process(inputFeature0)
            val outputFeature0 = outputs.outputFeature0AsTensorBuffer.floatArray

            outputFeature0.forEachIndexed { index, fl ->
                if (outputFeature0[maxIdx] <fl){
                    maxIdx = index
                }
            }
            if (maxIdx == 0){
                result.text = "Mild Demented"
                result.visibility = View.VISIBLE
            }
            if (maxIdx == 1){
                result.text = "Moderate Demented"
                result.visibility = View.VISIBLE
            }
            if (maxIdx == 3){
                result.text = "Non Demented"
                result.setTextColor(Color.GREEN)
                result.visibility = View.VISIBLE
            }
            if (maxIdx == 2){
                result.text = "Very Mild Demented"
                result.visibility = View.VISIBLE
            }

            model.close()

        }
        
    }


    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if(requestCode == REQUEST_CODE ){
            var uri = data?.data
            bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver , uri)
            imageView.setImageBitmap(bitmap)
        }
    }




}