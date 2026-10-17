// Local OCR helper. Uses only Apple's installed Vision and PDFKit frameworks.
import Foundation
import Vision
import PDFKit
import AppKit

struct Line: Codable {
    let text: String
    let confidence: Float
    let x: Double
    let y: Double
    let width: Double
    let height: Double
}
struct Result: Codable {
    let engine: String
    let revision: Int
    let languages: [String]
    let languageCorrection: Bool
    let osVersion: String
    let page: Int
    let pixelWidth: Int
    let pixelHeight: Int
    let lines: [Line]
    let text: String
}

do {
    let args = CommandLine.arguments
    guard args.count == 4, let pageNumber = Int(args[2]), pageNumber > 0,
          let pdf = PDFDocument(url: URL(fileURLWithPath: args[1])),
          let page = pdf.page(at: pageNumber - 1) else {
        throw NSError(domain: "OCR", code: 1, userInfo: [NSLocalizedDescriptionKey: "Usage: vision-pdf-ocr PDF PHYSICAL_PAGE OUTPUT_JSON"])
    }
    if FileManager.default.fileExists(atPath: args[3]) {
        throw NSError(domain: "OCR", code: 2, userInfo: [NSLocalizedDescriptionKey: "Output already exists"])
    }
    let bounds = page.bounds(for: .mediaBox)
    // 180 dpi at PDF's 72-points-per-inch coordinate scale.
    let target = NSSize(width: bounds.width * 2.5, height: bounds.height * 2.5)
    let bitmap = page.thumbnail(of: target, for: .mediaBox)
    var rect = CGRect(origin: .zero, size: bitmap.size)
    guard let image = bitmap.cgImage(forProposedRect: &rect, context: nil, hints: nil) else {
        throw NSError(domain: "OCR", code: 3, userInfo: [NSLocalizedDescriptionKey: "Cannot rasterize page"])
    }
    let request = VNRecognizeTextRequest()
    request.revision = VNRecognizeTextRequestRevision3
    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = false
    let supported = try request.supportedRecognitionLanguages()
    let languages = ["es-ES", "en-US"].filter { supported.contains($0) }
    guard languages.contains("es-ES") else {
        throw NSError(domain: "OCR", code: 4, userInfo: [NSLocalizedDescriptionKey: "Spanish recognition unavailable"])
    }
    request.recognitionLanguages = languages
    request.minimumTextHeight = 0.003
    try VNImageRequestHandler(cgImage: image, options: [:]).perform([request])
    let observations = request.results ?? []
    // Preserve native Vision order; don't manufacture a table reconstruction.
    let lines: [Line] = observations.compactMap { observation in
        guard let candidate = observation.topCandidates(1).first else { return nil }
        let box = observation.boundingBox
        return Line(text: candidate.string, confidence: candidate.confidence,
                    x: box.origin.x, y: box.origin.y, width: box.width, height: box.height)
    }
    let result = Result(engine: "Apple Vision VNRecognizeTextRequest", revision: request.revision,
                        languages: languages, languageCorrection: request.usesLanguageCorrection,
                        osVersion: ProcessInfo.processInfo.operatingSystemVersionString,
                        page: pageNumber, pixelWidth: image.width, pixelHeight: image.height,
                        lines: lines, text: lines.map { $0.text }.joined(separator: "\n"))
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
    try encoder.encode(result).write(to: URL(fileURLWithPath: args[3]), options: .withoutOverwriting)
} catch {
    FileHandle.standardError.write(Data("\(error)\n".utf8))
    exit(1)
}
