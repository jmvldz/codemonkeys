use anyhow::{Result, Context};
use regex::Regex;
use serde_json::Value;

/// Extract the last JSON array or object from a string
pub fn extract_last_json(text: &str) -> Result<Vec<String>> {
    // Look for the last JSON array in the text, which should be surrounded by ```
    let re = Regex::new(r#"```\s*\[\s*(?:"[^"]*"\s*,?\s*)+\]\s*```"#).unwrap();
    
    let json_str = if let Some(captures) = re.find(text) {
        // Extract just the JSON part (remove the ```)
        let json_with_backticks = &text[captures.start()..captures.end()];
        let json_re = Regex::new(r#"```\s*(\[.*\])\s*```"#).unwrap();
        if let Some(json_match) = json_re.captures(json_with_backticks) {
            json_match.get(1).map(|m| m.as_str()).unwrap_or_default()
        } else {
            return Err(anyhow::anyhow!("Failed to extract JSON from backticks"));
        }
    } else {
        // Try without the backticks if we didn't find anything
        let re = Regex::new(r#"\[\s*(?:"[^"]*"\s*,?\s*)+\]"#).unwrap();
        if let Some(captures) = re.find(text) {
            &text[captures.start()..captures.end()]
        } else {
            return Err(anyhow::anyhow!("No JSON array found in text"));
        }
    };
    
    // Parse the JSON string as a Value
    let json_value: Value = serde_json::from_str(json_str)
        .context("Failed to parse JSON")?;
    
    // Extract string values from the array
    if let Value::Array(array) = json_value {
        let string_values = array
            .into_iter()
            .filter_map(|val| {
                if let Value::String(s) = val {
                    Some(s)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        
        Ok(string_values)
    } else {
        Err(anyhow::anyhow!("JSON value is not an array"))
    }
}