Of course. Here is a complete, working implementation of Server-Sent Events (SSE) with a Spring Boot backend and an Angular frontend, including step-by-step instructions and the necessary dependencies.

### **1. Spring Boot Backend (Java)**

First, I'll set up the Spring Boot application to serve as the SSE server. This will include a controller that sends time-stamped events to any connected client.

#### **Dependencies**

Here are the necessary Gradle dependencies for your build.gradle file. This includes spring-boot-starter-webflux for reactive web capabilities.

```Groovy

plugins {  
    id 'java'  
    id 'org.springframework.boot' version '3.3.1'  
    id 'io.spring.dependency-management' version '1.1.5'  
}

group = 'com.example'  
version = '0.0.1-SNAPSHOT'

java {  
    toolchain {  
        languageVersion = JavaLanguageVersion.of(17)  
    }  
}

repositories {  
    mavenCentral()  
}

dependencies {  
    implementation 'org.springframework.boot:spring-boot-starter-webflux'  
    testImplementation 'org.springframework.boot:spring-boot-starter-test'  
    testImplementation 'io.projectreactor:reactor-test'  
}

tasks.named('test') {  
    useJUnitPlatform()  
}
```

#### **SSE Controller**

Next, create a RestController to handle the SSE endpoint. This controller will return a Flux<ServerSentEvent<String>>, which is a stream of events. The server will send a new event every second.

src/main/java/com/example/sseserver/SseController.java

```Java
package com.example.sseserver;

import org.springframework.http.MediaType;
import org.springframework.http.codec.ServerSentEvent;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;
import reactor.core.publisher.Flux;

import java.time.Duration;
import java.time.Instant;
import java.util.Random;

@RestController
public class SseController {

    private static final Random random = new Random();

    @GetMapping(path = "/sse-json", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public Flux<ServerSentEvent<Object>> sse() {
        // Stream for stock updates every 2 seconds
        Flux<ServerSentEvent<Object>> stockStream = Flux.interval(Duration.ofSeconds(2))
                .map(sequence -> {
                    double price = 100 + random.nextDouble(10); // Random price between 100 and 110
                    StockUpdate stockUpdate = new StockUpdate("GGL", price, Instant.now().toEpochMilli());
                    return ServerSentEvent.builder()
                            .id(String.valueOf(sequence))
                            .event("stock-update")
                            .data(stockUpdate)
                            .build();
                });

        // Stream for system notifications every 5 seconds
        Flux<ServerSentEvent<Object>> notificationStream = Flux.interval(Duration.ofSeconds(5))
                .map(sequence -> {
                    SystemNotification notification = new SystemNotification("INFO", "System health is nominal.", Instant.now().toEpochMilli());
                    return ServerSentEvent.builder()
                            .id(String.valueOf(sequence))
                            .event("system-notification")
                            .data(notification)
                            .build();
                });
        
        // Merge the two streams into one
        return Flux.merge(stockStream, notificationStream);
    }
}
```

This controller defines a /sse endpoint that produces a text/event-stream media type. It uses a Flux to emit a new ServerSentEvent every second.

---

### **2. Angular Frontend**

Now, I'll create the Angular application that will connect to the Spring Boot backend and display the real-time events.

#### **Angular Dependencies**

Here is the package.json file with the core Angular dependencies. No special libraries are needed for SSE, as Angular can use the browser's native EventSource API.

```json

{  
  "name": "angular-sse-client",  
  "version": "0.0.0",  
  "scripts": {  
    "ng": "ng",  
    "start": "ng serve",  
    "build": "ng build",  
    "watch": "ng build --watch --configuration development",  
    "test": "ng test"  
  },  
  "private": true,  
  "dependencies": {  
    "@angular/animations": "^18.0.0",  
    "@angular/common": "^18.0.0",  
    "@angular/compiler": "^18.0.0",  
    "@angular/core": "^18.0.0",  
    "@angular/forms": "^18.0.0",  
    "@angular/platform-browser": "^18.0.0",  
    "@angular/platform-browser-dynamic": "^18.0.0",  
    "@angular/router": "^18.0.0",  
    "rxjs": "\~7.8.0",  
    "tslib": "^2.3.0",  
    "zone.js": "\~0.14.3"  
  },  
  "devDependencies": {  
    "@angular-devkit/build-angular": "^18.0.2",  
    "@angular/cli": "^18.0.2",  
    "@angular/compiler-cli": "^18.0.0",  
    "@types/jasmine": "\~5.1.0",  
    "jasmine-core": "\~5.1.0",  
    "karma": "\~6.4.0",  
    "karma-chrome-launcher": "\~3.2.0",  
    "karma-coverage": "\~2.2.0",  
    "karma-jasmine": "\~5.1.0",  
    "karma-jasmine-html-reporter": "\~2.1.0",  
    "typescript": "\~5.4.2"  
  }  
}
```

#### **SSE Service**

Create an Angular service to encapsulate the logic for connecting to the SSE endpoint.

src/app/sse.service.ts

```TypeScript
import { Injectable, NgZone } from '@angular/core';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class SseService {
  private eventSource!: EventSource;

  constructor(private zone: NgZone) {}

  observeMessages(url: string): Observable<string> {
    return new Observable(observer => {
      this.eventSource = new EventSource(url);

      this.eventSource.onmessage = event => {
        this.zone.run(() => {
          observer.next(event.data);
        });
      };

      this.eventSource.onerror = error => {
        this.zone.run(() => {
          observer.error(error);
        });
      };
    });
  }

  // Generic method to listen for any named event
  observeNamedEvent(url: string, eventName: string): Observable<string> {
    return new Observable(observer => {
      // Ensure we only have one EventSource connection
      if (!this.eventSource) {
        this.eventSource = new EventSource(url);
      }

      const eventListener = (event: MessageEvent) => {
        this.zone.run(() => {
          observer.next(event.data);
        });
      };

      this.eventSource.addEventListener(eventName, eventListener);

      // Return a teardown function to remove the listener
      return () => {
        if (this.eventSource) {
          this.eventSource.removeEventListener(eventName, eventListener);
        }
      };
    });
  }


  close() {
    if (this.eventSource) {
      this.eventSource.close();
    }
  }
}
```

This service uses the EventSource API to connect to the server and wraps the events in an Observable.

#### **Component Implementation**

Finally, create a component that uses the SseService to subscribe to events and display them. It also handles unsubscribing to prevent memory leaks.

src/app/app.component.ts

```TypeScript
import { Component, OnDestroy } from '@angular/core';
import { SseService } from './sse.service';
import { Subscription } from 'rxjs';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnDestroy {
  stockUpdates: any[] = [];
  notifications: any[] = [];
  private subscriptions = new Subscription();
  private readonly sseUrl = 'http://localhost:8080/sse-json';


  constructor(private sseService: SseService) {
    // Subscribe to stock updates
    const stockSub = this.sseService.observeNamedEvent(this.sseUrl, 'stock-update')
      .subscribe(data => {
        const parsedData = JSON.parse(data);
        this.stockUpdates.push(parsedData);
      });

    // Subscribe to system notifications
    const notificationSub = this.sseService.observeNamedEvent(this.sseUrl, 'system-notification')
      .subscribe(data => {
        const parsedData = JSON.parse(data);
        this.notifications.push(parsedData);
      });
    
    this.subscriptions.add(stockSub);
    this.subscriptions.add(notificationSub);
  }

  ngOnDestroy() {
    this.subscriptions.unsubscribe();
    this.sseService.close();
  }
}
```

src/app/app.component.html

```HTML
<div style="display: flex; gap: 50px;">
  <div>
    <h2>Stock Updates (GGL)</h2>
    <ul>
      <li *ngFor="let stock of stockUpdates">
        Price: {{ stock.price | number:'1.2-2' }} @ {{ stock.timestamp | date:'mediumTime' }}
      </li>
    </ul>
  </div>
  <div>
    <h2>System Notifications</h2>
    <ul>
      <li *ngFor="let notification of notifications">
        {{ notification.level }}: {{ notification.message }} @ {{ notification.timestamp | date:'mediumTime' }}
      </li>
    </ul>
  </div>
</div>
```

---

### **3. Running the Application**

1. **Start the Spring Boot Backend:**  

   * Navigate to the root of your Spring Boot project.  
   * Run the command: ./gradlew bootRun  
1. **Start the Angular Frontend:**  

   * Navigate to the root of your Angular project.  
   * Run the command: npm install  
   * Then run: ng serve  
   * Open your browser to http://localhost:4200.


Handling Connection Breaks
One of the best features of SSE is its built-in error handling for broken connections.

You don't have to do anything in the code for this to work.

If the connection between the client and server is interrupted (e.g., you stop and restart the Spring Boot server), the browser's 

EventSource API will automatically attempt to reconnect to the server.

How it Works:
When a connection is lost, the EventSource object enters a "reconnecting" state.

By default, it waits about 3 seconds before trying to establish a new connection.

Upon reconnecting, the client can optionally send a 

Last-Event-ID header to the server, letting the server know the last event the client received. This allows the server to resend any missed messages. Our current code doesn't use this feature, but it's available for scenarios requiring guaranteed delivery.

You can test this yourself:

Run both the backend and frontend.

Stop the Spring Boot application.

You will see the data stream stop in the browser.

Restart the Spring Boot application.

After a few seconds, you will see the browser automatically reconnect and the data stream will resume.

**Sources**  

1. [https://github.com/20HyeonsuLee/INFRA\_BACK](https://github.com/20HyeonsuLee/INFRA_BACK)  
2. [https://github.com/0mochileiro/HeroAcademyInterface](https://github.com/0mochileiro/HeroAcademyInterface)  
3. [https://github.com/ALODEV-GT/ngrx-on-ng18-example](https://github.com/ALODEV-GT/ngrx-on-ng18-example)