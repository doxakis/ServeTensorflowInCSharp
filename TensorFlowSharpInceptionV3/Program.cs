using System;
using System.IO;
using System.Linq;
using System.Text;
using TensorFlow;

namespace TensorFlowSharpInceptionV3
{
    class Program
    {
        static void Main(string[] args)
        {
			var projectFolder = @"..\..\..\..\";
			byte[] model = File.ReadAllBytes(projectFolder + @"model\inception_v3.pb");
			using (var graph = new TFGraph())
			{
				graph.Import(new TFBuffer(model));

				using (var session = new TFSession(graph))
				{
					var inputTensor = TFTensor.CreateString(File.ReadAllBytes(projectFolder + @"files\299x299.jpg"));
					var runner = session.GetRunner();
					
					TFOutput outputLayer = graph["DecodeJpeg"][0];
					TFOutput inputLayer = graph["DecodeJpeg/contents"][0];

					runner.AddInput(inputLayer, inputTensor).Fetch(outputLayer);
					var output = runner.Run();

					var result = output[0];
					var pixels = (byte[][][])result.GetValue(true);

					StringBuilder builder = new StringBuilder();
					foreach (var line in pixels)
					{
						foreach (var col in line)
						{
							foreach (var pixel in col)
							{
								builder.Append(pixel + " ");
							}
							builder.Append("\n");
						}
					}
					File.WriteAllText(projectFolder + @"results\tensorflowsharp.txt", builder.ToString());
				}
			}
			Console.WriteLine("Press any key...");
			Console.ReadLine();
		}
	}
}
