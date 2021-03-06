package declareextraction.main;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import declareextraction.constructs.Action;
import declareextraction.constructs.DeclareModel;
import declareextraction.constructs.TextModel;
import declareextraction.evaluation.Evaluator;
import declareextraction.io.GSReader;
import declareextraction.io.ResultsWriter;
import declareextraction.textprocessing.DeclareConstructor;
import declareextraction.textprocessing.Interrelation;
import declareextraction.textprocessing.TextParser;

public class DeclareExtractor {

	static final int START = 0;
	static final int END = 108;
	
	
	public static final String CONSTRAINT_FILE = "input/datacollection.csv";
	public static final String RESULTS_FILE = "output/results.csv";
	
	
	List<String> texts;
	List<DeclareModel> gsModels;
	List<DeclareModel> genModels;
	List<Object[]> evalresultsList;
	
	TextParser parser;
	private DeclareConstructor declareConstructor;
	

	public static void main(String[] args) {
		DeclareExtractor t = new DeclareExtractor();
		try {
			ArrayList<String> sentences = t.extractFromTxt("./nn_outputs/w2vec_constraint_sentences.txt");
			for (String sentence : sentences) {
				t.runSingleConstraint(sentence);
				System.out.println("\n");
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	private ArrayList<String> extractFromTxt(String path) throws IOException {
		BufferedReader br = new BufferedReader(new FileReader(path));
		ArrayList<String> sentences = new ArrayList<>();
		while(br.ready()) {
			String line = br.readLine();
			sentences.add(line);
		}
		br.close();
		return sentences;
	}
	
	private void runSingleConstraint(String text) {
		declareConstructor = new DeclareConstructor();
		
//		System.out.println("Parsing case: " + text);
//		TextModel textModel = parser.parseConstraintString(text);
//
//		System.out.println("Extracted actions:");
//		for (Action a : textModel.getActions()) {
//			System.out.println(a);
//		}

		DeclareModel dm = declareConstructor.convertToDeclareModel(text);
	}
	
	
	private void runApproachOnCSVFile(String csvInputPath) {
		parser = new TextParser();
		declareConstructor = new DeclareConstructor();
		Evaluator eval = new Evaluator();
		genModels = new ArrayList<DeclareModel>();
		evalresultsList = new ArrayList<Object[]>();
		
				
		loadCases(csvInputPath);
			
		
		int b = 0;
		for (int i = 0; i < texts.size(); i++) {
			
			if (b >= START && b <= END) {
				DeclareModel gsModel = gsModels.get(i);			
				
				if (true) {
//				if (gsModel.getConstraints().size() == 1) {

					String text = texts.get(i);
					System.out.println(i + " Parsing case: " + text);
					TextModel textModel = parser.parseConstraintString(text);

					System.out.println("Extracted actions:");
					for (Action a : textModel.getActions()) {
						System.out.println(a);
					}
					System.out.println("Extracted relations:");
					for (Interrelation rel : textModel.getInterrelations()) {
						System.out.println(rel);
					}

					DeclareModel dm = declareConstructor.convertToDeclareModel(textModel);
					dm.addTextModel(textModel);

					genModels.add(dm);

					Object[] evalres = eval.evaluateCase(dm, gsModel);
					evalresultsList.add(evalres);

				}
			}
			b++;
		}
		eval.printOverallResults();
		ResultsWriter.write(RESULTS_FILE, genModels, gsModels, evalresultsList);
	}
	
	
	private void loadCases(String csvInputPath) {
		gsModels = GSReader.loadGSFile(new File(csvInputPath));
		texts = new ArrayList<String>();
		for (DeclareModel dm : gsModels) {
			texts.add(dm.getText());
		}
	}

}
