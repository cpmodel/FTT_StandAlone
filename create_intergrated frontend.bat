
cd Manager_new
call npm run-script build
call xcopy "dist\*.*" "\frontend\*.*" /e /y
cd ..
