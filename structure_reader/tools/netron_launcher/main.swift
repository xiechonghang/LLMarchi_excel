// LLMarchi Netron 链接启动器（项目专用，最小实现）。
// 只接受 llmarchi-netron://open?candidate=<64hex>&phase=prefill|decode&module=<id>&variant=sim
// 仅解析到本项目候选的 model.sim.onnx，校验 run_manifest 中的 SHA256 后交给已安装 Netron。
import AppKit
import CryptoKit
import Foundation

let scheme = "llmarchi-netron"
let netronApp = URL(fileURLWithPath: "/Applications/Netron.app")
let phases: Set<String> = ["prefill", "decode"]
let modules: Set<String> = [
    "mhc_attn", "norm_attn", "attn_q", "attn_kv", "attn_cache", "attn_csa",
    "attn_indexer", "attn_mask", "attn_core", "attn_out", "res_attn",
    "mhc_ffn", "norm_ffn", "moe_route", "moe_expert", "moe_shared",
    "moe_merge", "res_ffn",
]

enum LinkError: Error, CustomStringConvertible {
    case message(String)
    var description: String {
        switch self { case .message(let m): return m }
    }
}

func projectRoot() throws -> URL {
    guard let raw = Bundle.main.object(forInfoDictionaryKey: "LLMarchiProjectRoot") as? String,
          !raw.isEmpty else {
        throw LinkError.message("Info.plist 缺少 LLMarchiProjectRoot")
    }
    let root = URL(fileURLWithPath: raw).standardizedFileURL.resolvingSymlinksInPath()
    var isDir: ObjCBool = false
    guard FileManager.default.fileExists(atPath: root.path, isDirectory: &isDir), isDir.boolValue else {
        throw LinkError.message("项目根不存在: \(root.path)")
    }
    let outputs = root.appendingPathComponent("structure_reader/outputs/deepseek_v4_flash_0731_hf")
    var outDir: ObjCBool = false
    guard FileManager.default.fileExists(atPath: outputs.path, isDirectory: &outDir), outDir.boolValue else {
        throw LinkError.message("候选输出目录不存在: \(outputs.path)")
    }
    return root
}

func isRegularFile(_ url: URL) -> Bool {
    guard let attrs = try? FileManager.default.attributesOfItem(atPath: url.path),
          let type = attrs[.type] as? FileAttributeType else { return false }
    return type == .typeRegular
}

func hexSHA256(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}

func resolve(_ link: String) throws -> (URL, String) {
    guard let comps = URLComponents(string: link) else {
        throw LinkError.message("URL 无法解析")
    }
    guard comps.scheme == scheme else {
        throw LinkError.message("scheme 必须是 \(scheme)")
    }
    guard comps.host == "open" else {
        throw LinkError.message("host 必须是 open")
    }
    guard comps.fragment == nil else {
        throw LinkError.message("不允许 fragment")
    }
    guard comps.user == nil, comps.password == nil else {
        throw LinkError.message("不允许 userinfo/password")
    }
    guard comps.port == nil else {
        throw LinkError.message("不允许 port")
    }
    guard comps.path.isEmpty || comps.path == "/" else {
        throw LinkError.message("不允许 path")
    }
    let items = comps.queryItems ?? []
    guard items.count == 4 else {
        throw LinkError.message("参数必须恰好 4 个（candidate/phase/module/variant）")
    }
    var params: [String: String] = [:]
    for item in items {
        guard ["candidate", "phase", "module", "variant"].contains(item.name) else {
            throw LinkError.message("未知参数: \(item.name)")
        }
        guard params[item.name] == nil else {
            throw LinkError.message("重复参数: \(item.name)")
        }
        guard let value = item.value, !value.isEmpty else {
            throw LinkError.message("参数为空: \(item.name)")
        }
        params[item.name] = value
    }
    guard let candidate = params["candidate"], candidate.range(of: "^[0-9a-f]{64}$", options: .regularExpression) != nil else {
        throw LinkError.message("candidate 必须是 64 位小写 hex")
    }
    guard let phase = params["phase"], phases.contains(phase) else {
        throw LinkError.message("phase 非法")
    }
    guard let module = params["module"], modules.contains(module) else {
        throw LinkError.message("module 非法")
    }
    guard let variant = params["variant"], variant == "sim" else {
        throw LinkError.message("variant 仅支持 sim")
    }
    let root = try projectRoot()
    let base = root.appendingPathComponent("structure_reader/outputs/deepseek_v4_flash_0731_hf/\(candidate)")
    guard FileManager.default.fileExists(atPath: base.path) else {
        throw LinkError.message("候选不存在: \(candidate)")
    }
    let baseResolved = base.resolvingSymlinksInPath().standardizedFileURL
    guard baseResolved.path == base.standardizedFileURL.path else {
        throw LinkError.message("候选目录含符号链接，拒绝")
    }
    let file = base.appendingPathComponent("onnx/\(phase)/\(module)/model.sim.onnx")
    let fileResolved = file.resolvingSymlinksInPath().standardizedFileURL
    guard fileResolved.path.hasPrefix(root.path + "/"), fileResolved.path == file.standardizedFileURL.path else {
        throw LinkError.message("路径越界或含符号链接，拒绝")
    }
    guard FileManager.default.fileExists(atPath: file.path), isRegularFile(file) else {
        throw LinkError.message("ONNX 不是常规文件: \(file.path)")
    }
    let manifest = base.appendingPathComponent("run_manifest.json")
    let manifestResolved = manifest.resolvingSymlinksInPath().standardizedFileURL
    guard manifestResolved.path.hasPrefix(root.path + "/"),
          manifestResolved.path == manifest.standardizedFileURL.path,
          FileManager.default.fileExists(atPath: manifest.path),
          isRegularFile(manifest) else {
        throw LinkError.message("run_manifest.json 不是项目内常规文件")
    }
    guard let manifestData = try? Data(contentsOf: manifest),
          let obj = try? JSONSerialization.jsonObject(with: manifestData) as? [String: Any],
          let deliverables = obj["deliverables"] as? [String: String] else {
        throw LinkError.message("run_manifest.json 不可读或缺少 deliverables")
    }
    let key = "onnx/\(phase)/\(module)/model.sim.onnx"
    guard let expected = deliverables[key] else {
        throw LinkError.message("run_manifest 无该交付: \(key)")
    }
    guard let data = try? Data(contentsOf: file) else {
        throw LinkError.message("无法读取 ONNX")
    }
    let actual = hexSHA256(data)
    guard actual == expected else {
        throw LinkError.message("SHA256 不匹配: manifest=\(expected) actual=\(actual)")
    }
    return (file, actual)
}

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(("error: " + message + "\n").data(using: .utf8)!)
    exit(1)
}

// --resolve-only <url>：只解析/校验，不启动 UI、不打开 Netron（测试用）
let args = CommandLine.arguments
if let idx = args.firstIndex(of: "--resolve-only") {
    guard idx + 1 < args.count else { fail("--resolve-only 需要 URL 参数") }
    do {
        let (url, sha) = try resolve(args[idx + 1])
        print("resolved: \(url.path)")
        print("sha256: \(sha)")
        exit(0)
    } catch {
        fail("\(error)")
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    private var urlReceived = false
    private var exitTimer: DispatchWorkItem?

    func application(_ application: NSApplication, open urls: [URL]) {
        urlReceived = true
        exitTimer?.cancel()
        exitTimer = nil
        for url in urls {
            do {
                let (file, _) = try resolve(url.absoluteString)
                let config = NSWorkspace.OpenConfiguration()
                // 等 NSWorkspace completion（Netron 实际接收/启动完成）后再退出
                NSWorkspace.shared.open([file], withApplicationAt: netronApp, configuration: config) { _, error in
                    if let error {
                        self.showError("打开 Netron 失败: \(error.localizedDescription)")
                    }
                    NSApp.terminate(nil)
                }
            } catch {
                showError("\(error)")
                NSApp.terminate(nil)
            }
            return
        }
        NSApp.terminate(nil)
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        // 仅“无 URL 启动”走短延时退出；收到 URL 时取消，绝不在 Netron 启动期间杀进程
        let work = DispatchWorkItem { [weak self] in
            guard self?.urlReceived == false else { return }
            NSApp.terminate(nil)
        }
        exitTimer = work
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.5, execute: work)
    }

    func showError(_ message: String) {
        let alert = NSAlert()
        alert.messageText = "LLMarchi Netron 链接"
        alert.informativeText = message
        alert.alertStyle = .warning
        alert.runModal()
    }
}

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.setActivationPolicy(.accessory)
app.run()
