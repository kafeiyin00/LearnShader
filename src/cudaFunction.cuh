

void __declspec(dllexport) calculateNormal(float* depth, int width, int height, float* nx,float* ny,float* nz);

void __declspec(dllexport) calculatePlanarity(float* depth, int width, int height, float* planarity);

void __declspec(dllexport) getCudaState();