#include<bits/stdc++.h>
#include<windows.h>
using namespace std;
int main(){
	srand(time(0));
	int money=1000;
	int num=0;
	
	for(int t=0;money>100;t++){
		cout<<"��"<<t<<"���G"; 
		money=money-100;
		if(rand()%100<num){
			cout<<"���F";
			break;
		}
		num=num+10;
		cout<<"�S��"<<endl;
		Sleep(500);
	}
}

/*
#include <unistd.h>
sleep(1);
*/
