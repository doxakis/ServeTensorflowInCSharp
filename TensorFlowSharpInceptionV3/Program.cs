using System;
using System.IO;
using System.Linq;
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
					var inputTensor = TFTensor.CreateString(File.ReadAllBytes(projectFolder + @"files\cat1.jpg"));
					var runner = session.GetRunner();

					TFOutput classificationLayer = graph.Cast(graph["softmax"][0], TFDataType.Double);
					TFOutput bottleneckLayer = graph["pool_3"][0];

					TFOutput tIn = graph["DecodeJpeg/contents"][0];
					runner.AddInput(tIn, inputTensor).Fetch(classificationLayer, bottleneckLayer);
					var output = runner.Run();

					var result = output[0];
					var probabilities = ((double[][])result.GetValue(jagged: true))[0];

					File.WriteAllText(projectFolder + @"results\tensorflowsharp.txt", string.Join(Environment.NewLine, probabilities.Select(m => m.ToString("0.00000000000000000"))));
				}
			}
			Console.WriteLine("Press any key...");
			Console.ReadLine();
		}
	}
}
