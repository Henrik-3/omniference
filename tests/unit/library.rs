use omniference::router::AdapterRegistry;

#[test]
fn library_exposes_an_empty_adapter_registry() {
	assert!(AdapterRegistry::default().is_empty());
}
