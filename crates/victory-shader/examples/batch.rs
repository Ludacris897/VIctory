use victory_shader::*;
fn main() {
    let t = emit::Target { lanes_min: 32, lanes_max: 32 };
    let mut ok = 0; let mut bad = 0;
    for path in std::env::args().skip(1) {
        let bytes = std::fs::read(&path).unwrap();
        let name = std::path::Path::new(&path).file_stem().unwrap().to_string_lossy().to_string();
        match std::panic::catch_unwind(|| emit::translate_for(&bytes, t)) {
            Ok(Ok(_)) => { ok += 1; println!("OK   {name}"); }
            Ok(Err(e)) => { bad += 1; println!("FAIL {name}: {e}"); }
            Err(_) => { bad += 1; println!("PANIC {name}"); }
        }
    }
    println!("ok {ok} failed {bad}");
}
