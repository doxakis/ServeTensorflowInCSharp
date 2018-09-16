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
					// Convert the image in filename to a Tensor suitable as input to the Inception model.
					TFTensor tensor;
					{
						var inputTensor = TFTensor.CreateString(File.ReadAllBytes(projectFolder + @"files\cat1.jpg"));
						TFGraph graphImage3 = new TFGraph();
						TFOutput input = graphImage3.Placeholder(TFDataType.String);
						TFOutput output = graphImage3.DecodeJpeg(contents: input, channels: 3);
						using (var session2 = new TFSession(graphImage3))
						{
							tensor = session2.Run(
								inputs: new[] { input },
								inputValues: new[] { inputTensor },
								outputs: new[] { output })[0];
						}
					}

					{
						var runner = session.GetRunner();

						TFOutput classificationLayer = graph["softmax"][0];
						TFOutput bottleneckLayer = graph["pool_3"][0];

						TFOutput tIn = graph["DecodeJpeg"][0];
						runner.AddInput(tIn, tensor).Fetch(classificationLayer, bottleneckLayer);
						var output = runner.Run();
						
						var result = output[0];
						var probabilities = ((float[][])result.GetValue(jagged: true))[0];

						File.WriteAllText(projectFolder + @"results\tensorflowsharp.txt", string.Join(Environment.NewLine, probabilities.Select(m => m.ToString("0.00000000000000000"))));
					}
				}
			}
			Console.WriteLine("Press any key...");
			Console.ReadLine();
		}
	}
}
