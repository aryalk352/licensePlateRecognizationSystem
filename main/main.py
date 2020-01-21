import importlib, importlib.util


logicView = "/mnt/c/Users/kitna/Desktop/college_routine/fourth_semester/Deep_Learning/assignment3/main/MainProject.py"

def module_from_file(module_name, file_path):
	spec = importlib.util.spec_from_file_location(module_name, file_path)
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return module
	
mainProject = module_from_file("MainProject.py",logicView)
mainProject.test();
print("testtestadgadsfasd")