call npm run-script build
call xcopy "dist\*.*" "..\frontend\*.*" /e /y
