package de.tu_darmstadt.rs.synbio.common.circuit;

import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.deser.std.StdDeserializer;
import org.jgrapht.io.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;

public class CircuitDeserializer extends StdDeserializer<Circuit> {

    private static final Logger logger = LoggerFactory.getLogger(CircuitDeserializer.class);

    private int supergateCounter = 0;

    public CircuitDeserializer(Class<Circuit> t) {
        super(t);
    }

    @Override
    public Circuit deserialize(JsonParser jsonParser, DeserializationContext deserializationContext) throws IOException, JsonProcessingException {

        Circuit circuit = new Circuit("supergate_" + supergateCounter);
        supergateCounter ++;

        Gate.GateProvider gateProv = new Gate.GateProvider();
        Wire.WireProvider wireProv = new Wire.WireProvider();

        JSONImporter<Gate, Wire> importer = new JSONImporter<Gate, Wire>(gateProv, wireProv);

        String jsonContent = jsonParser.readValueAsTree().toString();
        StringReader reader = new StringReader(jsonContent);

        try {
            importer.importGraph(circuit, reader);
        } catch (ImportException e) {
            e.printStackTrace();
        }

        return circuit;
    }

    public Circuit deserializeString(String content) throws IOException, JsonProcessingException {

        Circuit circuit = new Circuit("circuit_" + supergateCounter);
        supergateCounter ++;

        Gate.GateProvider gateProv = new Gate.GateProvider();
        Wire.WireProvider wireProv = new Wire.WireProvider();

        JSONImporter<Gate, Wire> importer = new JSONImporter<Gate, Wire>(gateProv, wireProv);

        try {
            importer.importGraph(circuit, new StringReader(content));
        } catch (ImportException e) {
            e.printStackTrace();
        }

        return circuit;
    }
}
