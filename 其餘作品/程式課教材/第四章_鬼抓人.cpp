#include<bits/stdc++.h>
using namespace std;

int main(){
	srand(time(0));
	char map[100][100];
	cout<<"輸入地圖大小:";
	int m;cin>>m;
	for(int t=0;t<m;t++)for(int r=0;r<m;r++)map[t][r]=' ';	
	for(int t=0;t<10;t++){
		int txs=rand()%5+1;int tys=rand()%5+1;
		int tx=rand()%m;int ty=rand()%m;
		for(int a=0;a<txs;a++)for(int s=0;s<tys;s++)map[a+tx][s+ty]='X';			
	}
	int px=rand()%m;int py=rand()%m;
	int gx=rand()%m;int gy=rand()%m;
	if(map[px][py]=='X'){
		px=rand()%m;
		py=rand()%m;
	}
	if(map[gx][gy]=='X'){
		gx=rand()%m;
		gy=rand()%m;
	}
	map[px][py]='P';map[gx][gy]='G';
	for(;;){
		for(int t=0;t<m;t++){
			for(int r=0;r<m;r++)cout<<map[t][r]<<' ';
			cout<<endl;
		}
		cout<<endl;
		cout<<"(玩家)輸入移動方向[W/A/S/D]:";
		char way;cin>>way;
		map[px][py]=' ';
		if(way=='W')if(map[px-1][py]!='X')px--;
		if(way=='A')if(map[px][py-1]!='X')py--;
		if(way=='S')if(map[px+1][py]!='X')px++;
		if(way=='D')if(map[px][py+1]!='X')py++;
		map[px][py]='P';
		cout<<endl;
		cout<<"(鬼鬼)輸入移動方向[W/A/S/D]:";
		cin>>way;
		map[gx][gy]=' ';
		if(way=='W')if(map[gx-2][gy]!='X')gx=gx-2;
		if(way=='A')if(map[gx][gy-2]!='X')gy=gy-2;
		if(way=='S')if(map[gx+2][gy]!='X')gx=gx+2;
		if(way=='D')if(map[gx][gy+2]!='X')gy=gy+2;
		map[gx][gy]='G';
		if(px==gx&&py==gy){
			cout<<endl<<"呵呵...被抓到了..."<<endl; 
		}
		system("cls");
	}
}
