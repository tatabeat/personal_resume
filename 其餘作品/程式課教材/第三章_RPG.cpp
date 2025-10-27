#include<bits/stdc++.h>
using namespace std;
int main(){
	srand(time(0));
	int BHP=10000;
	int BATK=200;
	int PHP=1000;
	int PATK=100;
	string Bskill[3]={"拔刀斬","能量噬取","宿願"};
	string Pskill[3]={"偷襲","劍氣","居合斬"};
	for(;;){
		
		cout<<"Boss HP:"<<BHP<<endl;
		cout<<"Player HP:"<<PHP<<endl;
		cout<<endl<<"普通攻擊[A] / 技能[S] / 回復[D] / 逃跑[F] :";
		char choise;cin>>choise; 
		cout<<endl<<"---------------------------------"<<endl;
		if(choise=='A'){
			cout<<"你對Boss使用普通攻擊，造成"<<PATK<<"的傷害"<<endl; 
			BHP=BHP-PATK; 
		}
		if(choise=='S'){
			cout<<"偷襲[A] / 劍氣[S] / 居合斬[D] :";
			char skill;cin>>skill;
			if(skill=='A'){
				cout<<"選擇偷襲!"<<endl; 
				if(rand()%100<50){
					cout<<"偷襲成功，命中弱點造成"<<PATK*2<<"的傷害"<<endl; 
					BHP=BHP-PATK*2;
				}
				else cout<<"偷襲失敗了..."<<endl; 
			}
			if(skill=='S'){
				cout<<"對Boss斬出劍氣"<<endl; 
				BHP=BHP-PATK*0.8;
				if(rand()%100<30){
					cout<<"魔王大意了沒有閃，造成"<<PATK*2.2<<"點傷害"<<endl; 
					BHP=BHP-PATK*1.4;
				}
				else cout<<"魔王檔下劍氣但還是受到"<<PATK*0.8<<"點傷害"<<endl; 
			}
			if(skill=='D'){
				cout<<"一心一念，居合斬"<<endl; 
				if(rand()%100<10){
					cout<<"銳利的一擊，造成"<<PATK*10<<"點傷害"<<endl; 
					BHP=BHP-PATK*10;
				}
				else cout<<"很好，沒有打到"<<endl; 
			}	
		}
		if(choise=='D'){
			cout<<"治療自己"<<endl; 
			PHP=PHP+rand()%300+200;	
		} 
		if(choise=='F'){
			cout<<"你選擇逃跑...";
			return 0;
		}  
		if(rand()%100<65){
			cout<<"魔王揮刀，給予"<<BATK<<"點傷害"<<endl;
			PHP=PHP-BATK; 
		}
		else {
			int skill=rand()%3;
			if(skill==1){
				cout<<"魔王使出拔刀斬，造成"<<BATK+PHP*0.05<<"點傷害"<<endl;
				PHP=PHP-BATK-PHP*0.05;
			}	
			if(skill==2){
				cout<<"魔王噬取你的血液，吸血"<<BATK+BHP*0.01<<"點"<<endl;
				BHP=BHP+BATK+BHP*0.01;
				PHP=PHP-BATK-BHP*0.01; 
			}
			if(skill==3){
				cout<<"魔王發動【血願】，以血換血";
				BHP=BHP*0.5;
				PHP=PHP-BATK*BHP/1000;
				cout<<"....受到"<<BATK*BHP/1000<<"點傷害"<<endl; 
			}
		}
		if(BHP<=0){
			cout<<"魔王落敗，你贏了"<<endl;
			break; 
		}
		if(PHP<=0){
			cout<<"最後還是無法存活..."<<endl;
			break; 
		}
	} 
} 
